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

using matlab::mex::ArgumentList;
using namespace matlab::engine;
using namespace matlab::data;

std::ostringstream stream;

struct Options {
public:
	double threshold			= 0;
	double minIntensity			= 0;
	double maxIntensity			= 0;
	double scale				= 1;
	double viewOffset			= 0;
	size_t imageWidth			= 0;
	size_t imageHeight			= 0;
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

/* 3D VECTOR CLASS */
class Vector3 {
public:
	double x, y, z;
	Vector3() { x = y = z = 0; }
	Vector3(double scalar) : x(scalar), y(scalar), z(scalar){}
	Vector3(double i, double j, double k) : x(i), y(j), z(k) {}

	const double& operator [] (size_t i) const { return (&x)[i]; }
	double& operator [] (size_t i) { return (&x)[i]; }

	Vector3 operator + (Vector3 v) { return Vector3( x + v.x, y + v.y, z + v.z ); }
	Vector3 operator - (Vector3 v) { return Vector3( x - v.x, y - v.y, z - v.z ); }
	Vector3 operator * (double scalar) {
		return Vector3(x*scalar, y*scalar, z*scalar);
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
		Vector3 tmp(0, 1, 0);
		this->from = from;
		this->to = to;
		forward = (to - from).normalize();
		right = tmp.cross(forward);
		up = forward.cross(right);

		setupLookAtMatrix();
	}
	void yaw(double y) {
		double theta = degToRad(y);
		Matrix4x4 r = Matrix4x4(std::cos(theta), 0, std::sin(theta), 0,
			0, 1, 0, 0,
			-1 * std::sin(theta), 0, std::cos(theta), 0,
			0, 0, 0, 1);
		lookAt =  lookAt.matrixMulti(r);
	}
	void pitch(double p) {
		double theta = degToRad(p);
		Matrix4x4 r = Matrix4x4(1, 0, 0, 0,
			0, std::cos(theta), -1 * std::sin(theta), 0,
			0, std::sin(theta), std::cos(theta), 0,
			0, 0, 0, 1);
		lookAt = lookAt.matrixMulti(r);
	}
private:
	void setupLookAtMatrix() {
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

double interpolation(double x, double min, double max) {
	return  (x - min) * ((255) / (max - min));
}

Vector3 rasterToScreen(size_t w, size_t h,Options options) {
	// from raster to normalized to normalized to screen to camera
	double cameraX = (2 * (w + 0.5) / options.imageWidth - 1) * options.scale * options.imageAspectRatio;
	double cameraY = (1 - 2 * (h + 0.5) / options.imageHeight) * options.scale;

	return Vector3(cameraX, cameraY, -1);
}

class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
	//std::ostringstream stream;
	Options options;

public:
	void operator()(ArgumentList outputs, ArgumentList inputs) {
		checkArguments(inputs);

		const TypedArray<double> vol_size	= inputs[0];
		const TypedArray<double> volume		= inputs[1];
		const TypedArray<double> viewSize	= inputs[2];

		options.setImageSize(viewSize[0], viewSize[1]);
		options.setIntensity(inputs[3][0], inputs[4][0]);
		options.threshold = inputs[5][0];
		options.fov = 52.51;
		options.viewOffset = 100;
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
	
		Camera camera(Vector3(0, vol_size[1] + options.viewOffset, 0), //Vector3(vol_size[0]/2, vol_size[1]/2, vol_size[2]/2));
			Vector3(0, -1, 0));

		camera.yaw(rotation[0]);
		camera.pitch(rotation[1]);

		Vector3 rayStart = camera.lookAt.VecMatrixMulti(Vector3(0));

		int maxIteration = std::floor(std::fmax(vol_size[0], std::fmax(vol_size[1], vol_size[2])));
		int counterNoIntersection = 0;
		int counterIntersection = 0;

		/*stream << "grid" << grid << std::endl;
		stream << camera.lookAt << std::endl;		
		stream << rayStart << std::endl;*/

		std::ofstream oflog("./info.log"); // DEBUG INFO 
		std::ofstream ofs("./out.ppm", std::ios::out | std::ios::binary); //DEBUG IMAGE

		for (size_t h = 0; h < options.imageHeight; ++h) {
			for (size_t w = 0; w < options.imageWidth; ++w) {
								
				Vector3 rayEnd =  rasterToScreen(w,h,options);

				rayEnd = camera.lookAt.VecMatrixMulti(rayEnd);
				//oflog << w << " " << h << " " << rayEnd << std::endl;
				//Ray ray(rayStart, rayEnd);
				Ray ray(Vector3(rayStart.x+w , rayStart.y, rayStart.z+h), Vector3( w,  -1, h));

				Hit hit = computeRayABBoxIntersection(ray, grid);

				if (!hit.isHit)
				{
					counterNoIntersection++;	
					viewOutput[w][h] = 0;

				} else {
					counterIntersection++;

					Vector3 start = ray.orig + ray.dir *hit.tmin;
					Vector3 end = ray.orig + ray.dir *hit.tmax;
					
					int ix = std::floor(start.x) != 0 ? std::floor(start.x ) - 1 : 0;
					int iy = std::floor(start.y) != 0 ? std::floor(start.y ) - 1 : 0;
					int iz = std::floor(start.z) != 0 ? std::floor(start.z ) - 1 : 0;
					

					int xDir = (ray.dir.x > 0) ? 1 : (ray.dir.x < 0) ? -1 : 0;
					int yDir = (ray.dir.y > 0) ? 1 : (ray.dir.y < 0) ? -1 : 0;
					int zDir = (ray.dir.z > 0) ? 1 : (ray.dir.z < 0) ? -1 : 0;

					Vector3 dir(xDir, yDir, zDir);

					int xBound = ((dir.x > 0) ? ix + 1 : ix);
					int yBound = ((dir.y > 0) ? iy + 1 : iy);
					int zBound = ((dir.z > 0) ? iz + 1 : iz);

					Vector3 invDir(1 / dir.x, 1 / dir.y, 1 / dir.z);

					double xt = (xBound - start.x) * invDir.x ;
					double yt =  (yBound - start.y) * invDir.y ;
					double zt = (zBound - start.z) * invDir.z ;

					double xDelta =  dir.x * invDir.x ;
					double yDelta =  dir.y * invDir.y ;
					double zDelta =  dir.z * invDir.z ;


					double xOut = (dir.x < 0) ? -1 : grid.bounds[1].x;
					double yOut = (dir.y < 0) ? -1 : grid.bounds[1].y;
					double zOut = (dir.z < 0) ? -1 : grid.bounds[1].z;

					double value = options.minIntensity;

					for (int i = 0; i < maxIteration; i++) {
						if (xt < yt && xt < zt) {

							double xx = ((dir.x > 0) ? ix + 1 : ix);
							double newT = (xx - start.x) *invDir.x;

							ix += dir.x;
							if (ix == xOut) break;
							xt += xDelta;

						}
						else if (yt < zt) {

							double yy = ((dir.y > 0) ? iy + 1 : iy);
							double newT = (yy - start.y) *invDir.y;

							iy += dir.y;
							if (iy == yOut) break;
							yt += yDelta;

						} else {

							double zz = ((dir.z > 0) ? iz + 1 : iz);
							double newT = (zz - start.z) *invDir.z;

							iz += dir.z;
							if (iz == zOut) break;
							zt += zDelta;

						}
						
						if (ix < 0 || iy < 0 || iz < 0 || ix > grid.bounds[1].x || iy > grid.bounds[1].y || iz > grid.bounds[1].z) {
							/*Vector3 voxelIndex = Vector3(ix, iy, iz); stream << voxelIndex << std::endl; displayOnMATLAB(stream);*/
							break;
						}
						
						//value = volume[ix][iy][iz];
						double tmp = volume[ix][iy][iz];
						if (tmp > value) {
							if (tmp >= options.threshold)
								value = tmp;
							else
								value = options.minIntensity;
						}
							
						// movimento costante
						// salta re voxel non 
					}
					viewOutput[w][h] = interpolation(value, options.minIntensity,options.maxIntensity);
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