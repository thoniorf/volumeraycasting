/* Volume
 * Volumetric data 
*/
#ifndef VOLUME
#define VOLUME
#endif // !VOLUME
#define _USE_MATH_DEFINES

#include "mex.hpp"
#include "mexAdapter.hpp"
#include <string>
#include <ctime>
#include <cmath>
#include <iostream>
#include <fstream>
#include <exception>

using matlab::mex::ArgumentList;
using namespace matlab::engine;
using namespace matlab::data;

std::ostringstream stream;
std::time_t timer;
int timeout = 300;
struct Options {
public:
	double threshold			= 0;
	double minIntensity			= 0;
	double maxIntensity			= 0;
	double scale				= 1;
	double viewOffset			= 1;
	size_t imageWidth			= 2;	
	size_t imageHeight			= 2;
	double fov					= 60;
	double imageAspectRatio		= 1;

	void setImageSize(double width,double height) {
		imageWidth = std::floor(width);
		imageHeight = std::floor(height);
	}

	void setIntensity(double min, double max) {
		minIntensity = min;
		maxIntensity = max;
	}

};

double degToRad(double d) {
	return d * M_PI / 180;
}

double fixZeroDoublePrecisionError(double x) {
	if (std::fabs(x) <= 5.00e-5)
		return 0.0;
	return x;
}
/* 3D VECTOR CLASS */
class Vector3 {
public:
	double x, y, z;
	Vector3() { x = y = z = 0; }
	Vector3(double scalar) : x(scalar), y(scalar), z(scalar) {}
	Vector3(double i, double j, double k) : x(i), y(j), z(k){}

	const double& operator [] (size_t i) const { return (&x)[i]; }
	double& operator [] (size_t i) { return (&x)[i]; }

	Vector3 operator + (Vector3 v) { return Vector3( x + v.x, y + v.y, z + v.z ); }
	Vector3 operator - (Vector3 v) { return Vector3( x - v.x, y - v.y, z - v.z ); }
	Vector3 operator * (double scalar) {
		return Vector3(x*scalar, y*scalar, z*scalar);
	}
	Vector3 operator = (Vector3 v) {
		x = fixZeroDoublePrecisionError(v.x);
		y = fixZeroDoublePrecisionError(v.y);
		z = fixZeroDoublePrecisionError(v.z);
		return Vector3(x, y, z);
	}

	bool operator != (Vector3 v) {
		return (v.x != x || v.y != y || v.z != z);
	}
		
	double norm() const { return x * x + y * y + z * z; }
	double length() const { return sqrt(norm()); }

	Vector3 normalize() {
		double n = norm();
		double xx = x, yy = y , zz = z ;
		if (n > 0) {
			double factor = 1 / sqrt(n);
			xx *= factor, yy *= factor, zz *= factor;
		}
		return Vector3(xx, yy, zz);
	}

	Vector3 cross(Vector3 v) {
		double cx, cy, cz;
		cx = y * v.z - z * v.y;
		cy = z * v.x - x * v.z;
		cz = x * v.y - y * v.x;
		return Vector3(cx, cy, cz);
	}
	friend Vector3 operator /(int scalar, Vector3 vec) {
		return Vector3(scalar / vec.x, scalar / vec.y, scalar / vec.z);
	}

	friend std::ostream& operator << (std::ostream &s, const Vector3 &v)
	{
		return s << "(" << v.x << " , " << v.y << " , " << v.z << ")";
	}

};

class Matrix4x4 {
	/* 
	x1 x2 x3 0
	y1 y2 y3 0
	z1 z2 z3 0
	tx ty tz 1
	
	L = T * R * S
	scale , rotate, translate
	*/
private:
	double x[4][4] = { {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };
public:
	Matrix4x4() {}
	Matrix4x4(Vector3 translation) {
		// translation component
		x[0][3] = translation.x;
		x[1][3] = translation.y;
		x[2][3] = translation.z;

	}
	Matrix4x4(double c00, double c01, double c02, double c03, double c10, double c11, double c12, double c13, double c20, double c21, double c22, double c23, double c30, double c31, double c32, double c33) {
		x[0][0] = c00;
		x[0][1] = c01;
		x[0][2] = c02;
		x[0][3] = c03;
		x[1][0] = c10;
		x[1][1] = c11;
		x[1][2] = c12;
		x[1][3] = c13;
		x[2][0] = c20;
		x[2][1] = c21;
		x[2][2] = c22;
		x[2][3] = c23;
		x[3][0] = c30;
		x[3][1] = c31;
		x[3][2] = c32;
		x[3][3] = c33;
	}
	const double* operator [] (uint8_t i) const { return x[i]; }
	double* operator [] (uint8_t i) { return x[i]; }
	
	Matrix4x4 matrixMulti(const Matrix4x4 m) {
		Matrix4x4 c;
		for (uint8_t i = 0; i < 4; ++i) {
			for (uint8_t j = 0; j < 4; ++j) {
				c[i][j] = x[i][0] * m[0][j] + x[i][1] * m[1][j] + x[i][2] * m[2][j] + x[i][3] * m[3][j];
			}
		}
		return c;
	}
	Vector3 VecMatrixMulti(const Vector3 src) {
		double a, b, c, w;

		a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0] + x[3][0];
		b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1] + x[3][1];
		c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2] + x[3][2];
		w = src.x * x[0][3] + src.y * x[1][3] + src.z * x[2][3] + x[3][3];
		return Vector3(a / w, b / w, c / w);
	}

	Vector3 DirMatrixMulti(const Vector3 src) const
	{
		double a, b, c;

		a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0];
		b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1];
		c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2];

		return Vector3(a, b, c);
	}
	friend std::ostream& operator << (std::ostream &s, const Matrix4x4 &m)
	{

		s << "[" << m[0][0] <<
			" " << m[0][1] <<
			" " << m[0][2] <<
			" " << m[0][3] << "\n" <<

			" " << m[1][0] <<
			" " << m[1][1] <<
			" " << m[1][2] <<
			" " << m[1][3] << "\n" <<

			" " << m[2][0] <<
			" " << m[2][1] <<
			" " << m[2][2] <<
			" " << m[2][3] << "\n" <<

			" " << m[3][0] <<
			" " << m[3][1] <<
			" " << m[3][2] <<
			" " << m[3][3] << "]";

		return s;
	}
};
/* AABBOX CLASS */
class Grid {
public:
	Vector3 bounds[2] = { 0 };
	Grid() {}
	Grid(const Vector3 &min, const Vector3 &max) {
		bounds[0] = min, bounds[1] = max;
	}
	friend std::ostream& operator << (std::ostream &s, const Grid &g)
	{

		s << "[ " << g.bounds[0] << std::endl
			<< g.bounds[1] << "]" << std::endl;

		return s;
	}
};

/* RAY CLASS
@Parameter
	Vector3 origin    : the ray origin point expressed as Vector3
	Vector3 direction : the ray 'end' point expressed as Vector3
@Attributes
	Vector3 orig   : as origin parameter
	Vector3 dir    : normalized vector of direction parameter
	Vector3 invDir : inverse of direction
*/
class Ray {
public:
	Vector3 orig, end, dir, length, invDir;
	Ray(const Vector3 &startPoint,const Vector3 &endPoint):orig(startPoint),end(endPoint) {
		length = end-orig;
		dir = length.normalize();
		invDir = 1 / dir;
	}
	friend std::ostream& operator << (std::ostream &s, const Ray &v)
	{

		s << "[ " << std::endl
			<< "origin: " << v.orig << std::endl
			<< "end: " << v.end << std::endl
			<< "dir: " << v.dir << std::endl
			<< "invDir: " << v.invDir << std::endl
			<< "]";

		return s;
	}
};

struct Hit {
	bool isHit = false;
	double tmin;
	double tmax;
};

class ScreenSpace {
public:
	size_t width;
	size_t height;
	ScreenSpace(const int w, const int h) { width = w; height = h; }
};

class Camera {
public:
	Vector3 center;
	Vector3 from;
	Vector3 to;
	Vector3 forward;
	Vector3 right;
	Vector3 up;
	Matrix4x4 lookAt;
	Matrix4x4 worldTransformation;

	Camera(){}
	Camera(Vector3 center) {
		this->center = center;
		worldTransformation = Matrix4x4(center);
		
	}
	Camera(Vector3 from, Vector3 to ) {
		this->from = from;
		this->to = to;
		
		setupLookAtMatrix();
	}
	void yaw(double angle) { // z-axis
		double theta = degToRad(angle);

		Matrix4x4 r = Matrix4x4(std::cos(theta), -1* std::sin(theta), 0, 0,
			std::sin(theta), std::cos(theta), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		from = r.VecMatrixMulti(from);
		setupLookAtMatrix();
	}
	void pitch(double angle) { //y-axis
		double theta = degToRad(angle);
		Matrix4x4 r = Matrix4x4(std::cos(theta), 0, std::sin(theta), 0,
			0, 1, 0, 0,
			-1*std::sin(theta), 0, std::cos(theta), 0,
			0, 0, 0, 1);
		from = r.VecMatrixMulti(from);
		setupLookAtMatrix();
	}
private:
	void setupLookAtMatrix() {
		Vector3 tmp(0, 1, 0);
		forward = (to - from).normalize();
		right = tmp.cross(forward);
		up = forward.cross(right);

		lookAt = Matrix4x4(
			right.x, right.y, right.z, 0,
			up.x, up.y, up.z, 0,
			forward.x, forward.y, forward.z, 0,
			from.x, from.y, from.z, 1);
	}
};
Hit computeRayABBoxIntersection(Ray ray, Grid grid) {
	Hit hit;
	double tmin, tmax, tminy, tmaxy, tminz, tmaxz;
	// "An efficient and robust ray-box intersection algorithm. Amy Williams et al.2004.
	
	if (ray.invDir.x >= 0) {
		tmin = (grid.bounds[0].x - ray.orig.x) * ray.invDir.x;
		tmax = (grid.bounds[1].x - ray.orig.x) * ray.invDir.x;
	}
	else {
		tmax = (grid.bounds[0].x - ray.orig.x) * ray.invDir.x;
		tmin = (grid.bounds[1].x - ray.orig.x) * ray.invDir.x;
	}

	if (ray.invDir.y >= 0) {
		tminy = (grid.bounds[0].y - ray.orig.y) * ray.invDir.y;
		tmaxy = (grid.bounds[1].y - ray.orig.y) * ray.invDir.y;
	}
	else {
		tmaxy = (grid.bounds[0].y - ray.orig.y) * ray.invDir.y;
		tminy = (grid.bounds[1].y - ray.orig.y) * ray.invDir.y;
	}
	
	hit.tmin = tmin;
	hit.tmax = tmax;

	if (tmin > tmaxy || tminy > tmax) return hit;

	if (tminy > tmin) tmin = tminy;
	if (tmaxy < tmax) tmax = tmaxy;
	
	if (ray.invDir.z >= 0) {
		tminz = (grid.bounds[0].z - ray.orig.z) * ray.invDir.z;
		tmaxz = (grid.bounds[1].z - ray.orig.z) * ray.invDir.z;
	}
	else {
		tmaxz = (grid.bounds[0].z - ray.orig.z) * ray.invDir.z;
		tminz = (grid.bounds[1].z - ray.orig.z) * ray.invDir.z;
	}

	hit.tmin = tmin;
	hit.tmax = tmax;

	if (tmin > tmaxz || tminz > tmax) return hit;

	if (tminz > tmin) tmin = tminz;
	if (tmaxz < tmax) tmax = tmaxz;
	
	hit.tmin = tmin;
	hit.tmax = tmax;
	
	hit.isHit = true;

	return hit;
}

double interpolation(double x,Options option) {
	if (x < option.threshold) return option.minIntensity;
	return  (x - option.minIntensity) * ((255) / (option.maxIntensity - option.minIntensity));
}
Vector3 rasterToScreen(size_t w, size_t h, Options options) {
	// from raster to normalized to normalized to screen to camera
	double cameraX = (2 * (w + 0.5) / options.imageWidth - 1) * options.scale * options.imageAspectRatio;
	double cameraY = (1 - 2 * (h + 0.5) / options.imageHeight) * options.scale;

	return Vector3(cameraX, cameraY, -1);
}

Vector3 rasterToScreen(size_t w, size_t h,double z,Options options) {
	// from raster to normalized to normalized to screen to camera
	double cameraX = (2 * (w + 0.5) / options.imageWidth - 1) * options.imageAspectRatio;//*options.scale;
	double cameraY = (1 - 2 * (h + 0.5) / options.imageHeight);//*options.scale;

	return Vector3(cameraX, cameraY, z);
}

class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
	//std::ostringstream stream;
	Options options;

public:
	void operator()(ArgumentList outputs, ArgumentList inputs) {
		checkArguments(inputs);
		std::time(&timer);
		const TypedArray<double> vol_size	= inputs[0];
		const TypedArray<double> volume		= inputs[1];
		const TypedArray<double> viewSize	= inputs[2];

		options.setImageSize(viewSize[0], viewSize[1]);
		options.setIntensity(inputs[3][0], inputs[4][0]);
		options.threshold = inputs[5][0];
		options.fov = 52.51;
		options.viewOffset = 200;
		options.scale = std::tan(degToRad((options.fov * 0.5)));
		options.imageAspectRatio = options.imageWidth / options.imageHeight;

		const TypedArray<double> rotation = inputs[6];
	
		TypedArray<double> viewOutput = factory.createArray<double>({ options.imageWidth,options.imageHeight }, { {0} });

		stream << "VOL: "	<<volume.getNumberOfElements() << std::endl;
		stream << "INT: "	<< options.minIntensity << " " << options.maxIntensity << std::endl;
		stream << "SIZE: "	<< vol_size[0] << " "<< vol_size[1]<<" "<< vol_size[2] << std::endl;
		stream << "VIEW: "	<< options.imageWidth << " " << options.imageHeight << std::endl;
		stream << "ROT: y-" << rotation[0] << " p-" << rotation[1] << " r-" << rotation[2] << std::endl;
		displayOnMATLAB(stream);
		
		Grid grid(Vector3(0), Vector3(vol_size[0], vol_size[1], vol_size[2]));
	
		Camera camera(Vector3(vol_size[0] / 2 + options.viewOffset, vol_size[1] / 2, vol_size[2] / 2 ), Vector3(vol_size[0]/2, vol_size[1]/2, vol_size[2]/2));

		camera.pitch(rotation[1]);
		camera.yaw(rotation[0]);

		//print lookat matrix

		int maxIteration = std::floor(std::fmax(vol_size[0], std::fmax(vol_size[1], vol_size[2])));
		int counterNoIntersection = 0;
		int counterIntersection = 0;

		stream << "grid" << grid << std::endl;
		/*stream << camera.lookAt << std::endl;		
		stream << rayStart << std::endl;*/

		std::ofstream oflog("./info.log"); // DEBUG INFO 
		std::ofstream ofs("./out.ppm", std::ios::out | std::ios::binary); //DEBUG IMAGE
		
		double littleStep = 0.2;

		double halfWidth = std::fabs(options.imageWidth / 2);
		double halfHeight = std::fabs(options.imageHeight / 2);

		for (size_t h = 0; h < options.imageHeight; ++h) {
			for (size_t w = 0; w < options.imageWidth; ++w) {

				//Vector3 rayStart = camera.lookAt.VecMatrixMulti(rasterToScreen(w,h, 0,options));
				//Vector3 rayEnd = camera.lookAt.VecMatrixMulti(rasterToScreen(w, h, 1, options));
				double x = w - halfWidth;
				double y = h - halfHeight;

				Vector3 rayStart = camera.lookAt.VecMatrixMulti(Vector3(x,y,0));
				Vector3 rayEnd = camera.lookAt.VecMatrixMulti(Vector3(x,y, 1));
				
				/*stream <<"rayStart: "<< rayStart << std::endl;
				stream <<"rayend: "<< rayEnd << std::endl;*/

				Ray ray(rayStart, rayEnd);

				Hit hit = computeRayABBoxIntersection(ray, grid);

				if (!hit.isHit)
				{
					counterNoIntersection++;	
					viewOutput[w][h] = 0;

				} else {
					counterIntersection++;

					Vector3 start = ray.orig + ray.dir *hit.tmin;
					Vector3 end = ray.orig + ray.dir *hit.tmax;

					displayOnMATLAB(stream);
					while (start != end ) {
						if  (time(NULL) - timer >= timeout)
						{
							stream << "TIMEOUT" << std::endl;
							break;
						}
						//aggiustare il floor per gli indici, se start è fuori ritorna un valore diverso da zero, ma maggiore di max-1, e quindi volume[start] è out of bound; maybe max(start,grid.max)
						int ix = std::floor(start.x) != 0 ? std::floor(start.x) - 1 : std::floor(start.x);
						int iy = std::floor(start.y) != 0 ? std::floor(start.y) - 1 : std::floor(start.y);
						int iz = std::floor(start.z) != 0 ? std::floor(start.z) - 1 : std::floor(start.z);

						if (ix < 0 || iy < 0 || iz < 0 || ix > grid.bounds[1].x-1 || iy > grid.bounds[1].y-1 || iz > grid.bounds[1].z-1) {
							//stream << start << std::endl;
							//stream << std::floor(start.x) << " " << std::floor(start.y) << " " << std::floor(start.z) << std::endl;
							//stream << ix << " " << iy << " " << iz << std::endl;
							//displayOnMATLAB(stream);
							break;
						}
						//add try-catch no 
						try {
							double value = volume[ix][iy][iz];
							if (value >= options.threshold) {
								viewOutput[w][h] = interpolation(value, options);
								break;
							}
						}
						catch (std::exception& e) {
							stream<<e.what()<<std::endl;
							stream << start << std::endl;
							stream << std::floor(start.x) << " " << std::floor(start.y) << " " << std::floor(start.z) << std::endl;
							stream << ix << " " << iy << " " << iz << std::endl;
							displayOnMATLAB(stream);
							return;
						}
						
						start = start + ray.dir*littleStep;
						
					}
				}
			}
		}

		stream << "no intersection " << counterNoIntersection << std::endl;
		stream << "intersection " << counterIntersection << std::endl;
		displayOnMATLAB(stream);

		/* DEBUG
		*  Save the output image in a file without going through matlab
		*/
		ofs << "P6\n" << options.imageWidth << " " << options.imageHeight << "\n255\n";
		for (uint32_t j = 0; j < options.imageHeight; ++j) {
			for (uint32_t i = 0; i < options.imageWidth; ++i) {
				char r = (char)(viewOutput[i][j]);
				char g = (char)(viewOutput[i][j]);
				char b = (char)(viewOutput[i][j]);
				ofs << r << g << b;
			}
		}

		ofs.close();
		oflog.close();
		/* END DEBUG*/

		outputs[0] = viewOutput;
	}
   
    void  checkArguments(ArgumentList inputs) {
		if (inputs[0].getType() != ArrayType::DOUBLE ||
			inputs[0].getNumberOfElements() != 3)
		{
			matlabPtr->feval(u"error", 0,
				std::vector<Array>({ factory.createScalar("Input must be a vector of three elements") }));
		}
		if (inputs[1].getType() != ArrayType::DOUBLE /*||
			/*inputs[1].getNumberOfElements() != 3*/)
		{
			matlabPtr->feval(u"error", 0,
				std::vector<Array>({ factory.createScalar("Input must be a volume") }));
		}
		if (inputs[2].getType() != ArrayType::DOUBLE ||
			inputs[2].getNumberOfElements() != 2)
		{
			matlabPtr->feval(u"error", 0,
				std::vector<Array>({ factory.createScalar("Input must be a vector of two elements") }));
		}

    }
    
	void displayOnMATLAB(std::ostringstream& stream) {
		// Pass stream content to MATLAB fprintf function
		matlabPtr->feval(u"fprintf", 0,
			std::vector<Array>({ factory.createScalar(stream.str()) }));
		// Clear stream buffer
		stream.str("");
	}


};