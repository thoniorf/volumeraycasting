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
inline 
double fixZeroDoublePrecisionError(double x) {
	if (std::fabs(x) <= 5.00e-5)
		return 0;
	return x;
}
/* 3D VECTOR CLASS */
class Vector3 {
public:
	double x, y, z;
	Vector3() { x = y = z = 0; }
	Vector3(double scalar) : x(scalar), y(scalar), z(scalar) {}
	Vector3(double i, double j, double k) : x(fixZeroDoublePrecisionError(i)), y(fixZeroDoublePrecisionError(j)), z(fixZeroDoublePrecisionError(k)){}

	const double& operator [] (size_t i) const { return (&x)[i]; }
	double& operator [] (size_t i) { return (&x)[i]; }

	Vector3 operator - () { return Vector3(-x, -y, -z); }
	Vector3 operator + (Vector3 v) { return Vector3( x + v.x, y + v.y, z + v.z ); }
	Vector3 operator - (Vector3 v) { return Vector3( x - v.x, y - v.y, z - v.z ); }
	Vector3 operator * (Vector3 v) { return Vector3( x * v.x, y * v.y, z * v.z ); }
	Vector3 operator * (double scalar) { return Vector3(x * scalar, y * scalar, z * scalar); }
	Vector3 operator / (double scalar) { return Vector3(x / scalar, y / scalar, z / scalar); }
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

	double dot(Vector3 v) {
		return x * v.x + y * v.y + z * v.z;
	}

	Vector3 cross(Vector3 v) {
		double cx, cy, cz;
		cx = y * v.z - z * v.y;
		cy = z * v.x - x * v.z;
		cz = x * v.y - y * v.x;
		return Vector3(cx, cy, cz);
	}
	
	friend Vector3 operator / (double scalar, Vector3 vec) {
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
	bool isOutsideGrid(int ix, int iy, int iz) {
		return ix < 0 || iy < 0 || iz < 0 || ix > bounds[1].x - 1 || iy > bounds[1].y - 1 || iz > bounds[1].z - 1;
	}

	bool isInsideGrid(int i, int j, int k) {
		return i > 0 && j > 0 && k > 0 && i < bounds[1].x - 1 && j < bounds[1].y - 1 && k < bounds[1].z - 1;
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
	Vector3 backward;
	Vector3 right;
	Vector3 up;
	Matrix4x4 lookAt;
	Matrix4x4 worldTransformation;
	Matrix4x4 rotationM;
	double theta, phi,sigma;
	Camera(){}
	Camera(Vector3 center) {
		this->center = center;
		worldTransformation = Matrix4x4(center);
		
	}
	Camera(Vector3 from, Vector3 to ) {
		this->from = from;
		this->to = to;
		this->rotationM = Matrix4x4();
		/*setupLookAtMatrix();*/
	}
	void yaw(double angle) { // z-axis
		//backward = from - to;

		theta = degToRad(std::fmod(angle,360));

		//stream << "backward: " << backward << std::endl;
		

		Matrix4x4 rZ = Matrix4x4(std::cos(theta), -1 * std::sin(theta), 0, 0,
			std::sin(theta), std::cos(theta), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		stream << "yaw angle: " << angle << " in rad: " << theta << std::endl;
		//backward = rZ.DirMatrixMulti(backward);
		//from = to + backward;

		////stream << "from: " << from<<std::endl;
		//setupLookAtMatrix();
		rotationM = rotationM.matrixMulti(rZ);
	}
	void pitch(double angle) { //y-axis
		/*backward = from - to;
*/
		phi = degToRad(std::fmod(angle, 360));
		Matrix4x4 rY = Matrix4x4(std::cos(phi), 0, -1*std::sin(phi), 0,
			0, 1, 0, 0,
			std::sin(phi), 0, std::cos(phi), 0,
			0, 0, 0, 1);

		stream << "pitch angle: " << angle << " in rad: " << phi << std::endl;
		//backward = rY.DirMatrixMulti(backward);

		//from = to + backward;

		//stream << "from: " << from << std::endl;
		//setupLookAtMatrix();
		rotationM = rotationM.matrixMulti(rY);
	}

	void roll(double angle) {
		sigma = degToRad(std::fmod(angle, 360));
		Matrix4x4 rX = Matrix4x4(1, 0, 0, 0,
			0, std::cos(theta), -1 * std::sin(theta), 0,
			0, std::sin(theta), std::cos(theta), 0,
			0, 0, 0, 1);
		rotationM = rotationM.matrixMulti(rX);
	}

	void setupLookAtMatrix() {
		backward = from - to;
		backward = rotationM.DirMatrixMulti(backward);
		from = to + backward;

		Vector3 tmp = rotationM.DirMatrixMulti(Vector3(0,1,0));
		forward = (to - from).normalize();
		stream << "forward " << forward << std::endl;
		/*if (forward.y == 1) {
			tmp = Vector3(1, 0, 0);
		}
		else if (forward.y == -1) {
			tmp = Vector3(-1, 0, 0);
		}
		else if (forward.y > 0 && forward.x > 0)
			tmp = Vector3(1, 0, 0);
		else if (forward.y <= 0 && forward.x > 0) {
			tmp = Vector3(0, -1, 0);
		}*/

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
Vector3 getGradient(int ix, int iy, int iz,const TypedArray<double>& volume) {
	// component-wise linear interpolation
	// simple average for gray vector
	double a, b, c;
	int xi = (int)(ix + 0.5);
	double xT = ix + 0.5 - xi;
	a = (volume[xi][iy][iz] - volume[xi - 1][iy][iz]) * (1.0 - xT) + (volume[xi + 1][iy][iz] - volume[xi][iy][iz]) * xT;
	
	int yi = (int)(iy + 0.5);
	double yT = iy + 0.5 - yi;
	b = (volume[ix][iy][iz] - volume[ix][iy-1][iz]) * (1.0 - yT) + (volume[ix][iy+1][iz] - volume[ix][iy][iz]) * yT;

	int zi = (int)(iz + 0.5);
	double zT = iz + 0.5 - zi;
	c = (volume[ix][iy][zi] - volume[ix][iy][zi-1]) * (1.0 - zT) + (volume[ix][iy][zi+1] - volume[ix][iy][zi]) * zT;

	//double gray = (a + b + c) / 3;
	//return Vector3(gray,gray,gray);
	return Vector3(a, b, c);
}

class MexFunction : public matlab::mex::Function {
    ArrayFactory factory;
    std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
	Options options;


public:
	void debugAffineTransformation() {
		const double theta = degToRad(45);
		const double phi = degToRad(45);
		const double sigma = degToRad(45);
		Matrix4x4 rZ = Matrix4x4(std::cos(theta), -1* std::sin(theta), 0, 0,
			std::sin(theta), std::cos(theta), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		Matrix4x4 rY = Matrix4x4(std::cos(phi), 0, std::sin(phi), 0,
			0, 1, 0, 0,
			-1*std::sin(phi), 0, std::cos(phi), 0,
			0, 0, 0, 1);
		Matrix4x4 rX = Matrix4x4(1, 0, 0, 0,
			0, std::cos(sigma), -1*std::sin(sigma), 0,
			0, std::sin(sigma), std::cos(sigma), 0,
			0, 0, 0, 1);
		Vector3 v1(1, 0, 0);
		Vector3 v2(0, 0, 1);
		Vector3 v3(0, 1, 0);

		Vector3 vz = rZ.DirMatrixMulti(v1);
		Vector3 vy = rY.DirMatrixMulti(vz);
		Vector3 vx = rX.DirMatrixMulti(vz);
		stream << "DEBUG AFFINE TRANSFORMATION" << std::endl;
		stream << vz << std::endl;
		stream << vy << std::endl;
		stream << vx << std::endl;

		// moltiplicato per una distanza restituisce backward
		Vector3 dir(0);
		dir.x = std::cosf(phi) * std::cosf(theta);
		dir.y = std::sinf(phi);
		dir.z = std::cosf(phi) * std::sinf(theta);
		dir = dir * 200;
		stream << "trig: " << dir<< std::endl;
		stream << "END DEBUG" << std::endl;
		displayOnMATLAB(stream);
	}

	void operator()(ArgumentList outputs, ArgumentList inputs) {
		std::time(&timer);
		//debugAffineTransformation();
		checkArguments(inputs);

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
	
		TypedArray<double> viewOutput = factory.createArray<double>({ options.imageWidth,options.imageHeight,3 });

		stream << "VOL: "	<< volume.getNumberOfElements() << std::endl;
		stream << "INT: "	<< options.minIntensity << " " << options.maxIntensity << std::endl;
		stream << "SIZE: "	<< vol_size[0] << " "<< vol_size[1]<<" "<< vol_size[2] << std::endl;
		stream << "VIEW: "	<< options.imageWidth << " " << options.imageHeight << std::endl;
		stream << "ROT: y-" << rotation[0] << " p-" << rotation[1] << " r-" << rotation[2] << std::endl;
		displayOnMATLAB(stream);
		
		Vector3 lightPosition(vol_size[0] / 2 + options.viewOffset, vol_size[1] / 2, vol_size[2] / 2);

		Vector3 ambientColor(0.5, 0.5, 0.5);
		Vector3 diffuseColor(0.6, 0.6, 0.6);
		Vector3 specularColor(0.7, 0.7, 0.7);
		double shininess = 16.0;

		Grid grid(Vector3(0), Vector3(vol_size[0], vol_size[1], vol_size[2]));
	
		Camera camera(Vector3(vol_size[0] / 2 + options.viewOffset, vol_size[1] / 2, vol_size[2] / 2 ), Vector3(vol_size[0]/2, vol_size[1]/2, vol_size[2]/2));
		
		camera.pitch(rotation[1]);
		camera.roll(rotation[2]);
		camera.yaw(rotation[0]);



		camera.setupLookAtMatrix();
		int maxIteration = std::floor(std::fmax(vol_size[0], std::fmax(vol_size[1], vol_size[2])));
		int counterNoIntersection = 0;
		int counterIntersection = 0;

		stream << camera.lookAt << std::endl;		
		
		double littleStep = 1;
		double maxStep = std::sqrt(vol_size[0] * vol_size[0] + vol_size[1] * vol_size[1] + vol_size[2] * vol_size[2]);

		double halfWidth = std::fabs(options.imageWidth / 2);
		double halfHeight = std::fabs(options.imageHeight / 2);

		for (size_t h = 0; h < options.imageHeight; ++h) {
			for (size_t w = 0; w < options.imageWidth; ++w) {

				double x = w - halfWidth;
				double y = h - halfHeight;

				Vector3 rayStart = camera.lookAt.VecMatrixMulti(Vector3(x,y,0));
				Vector3 rayEnd = camera.lookAt.VecMatrixMulti(Vector3(x,y, 1));
				
				Ray ray(rayStart, rayEnd);

				Hit hit = computeRayABBoxIntersection(ray, grid);

				if (!hit.isHit)
				{
					counterNoIntersection++;	
					viewOutput[w][h][0] = 0;
					viewOutput[w][h][1] = 0;
					viewOutput[w][h][2] = 0;

				} else {
					counterIntersection++;

					Vector3 start = ray.orig + ray.dir *hit.tmin;
					Vector3 end = ray.orig + ray.dir *hit.tmax;

					for (size_t t = 0; t < maxStep; t++) {

						if  (time(NULL) - timer >= timeout)
						{
							stream << "TIMEOUT" << std::endl;
							break;
						}
						int ix = std::floor(start.x) > 0 ? std::floor(start.x) - 1 : 0;
						int iy = std::floor(start.y) > 0 ? std::floor(start.y) - 1 : 0;
						int iz = std::floor(start.z) > 0 ? std::floor(start.z) - 1 : 0;

						if (grid.isInsideGrid(ix, iy, iz)) {
							if (volume[ix][iy][iz] >= options.threshold) {
								try {
									
									/*int up = iy + 1;
									int down = iy - 1;
									int front = iz + 1;
									int back = iz - 1;
									int left = ix + 1;
									int right = ix - 1;

									double up_voxel = volume[ix][up][iz];
									double down_voxel = volume[ix][down][iz];
									double front_voxel = volume[ix][iy][front];
									double back_voxel = volume[ix][iy][back];
									double left_voxel = volume[left][iy][iz];
									double right_voxel = volume[right][iy][iz];
									double sum = up_voxel + down_voxel + front_voxel + back_voxel + left_voxel + right_voxel;
									double interpolated = sum / 6;
									*/
									Vector3 grad = getGradient(ix, iy, iz, volume);
									Vector3 normal = -grad / std::sqrt(grad.norm());
									Vector3 lightDir = lightPosition.normalize();
									double distance = lightPosition.length();
									double lambertian = lightDir.dot(normal);

									Vector3 viewDir = -lightPosition.normalize();
									Vector3 halfDir = (lightDir + viewDir).normalize();
									double specAngle = std::fmax(halfDir.dot(normal), 0.0);
									double specular = std::pow(specAngle, shininess);
									//1* lightAmbientColor - Ambient component
									//1* lightDiffuseColor*dot(lightdir,normals)
									//1* lightSpecularColor * [dot(halfdir,normals)]^shininess
									Vector3 IlluminationI = ambientColor + diffuseColor * lambertian  +specularColor * specular;

									viewOutput[w][h][0] = IlluminationI.x;
									viewOutput[w][h][1] = IlluminationI.y;
									viewOutput[w][h][2] = IlluminationI.z;
									break;

								}
								catch (std::exception& e) {
									stream << e.what() << std::endl;
									stream << start << std::endl;
									stream << std::floor(start.x) << " " << std::floor(start.y) << " " << std::floor(start.z) << std::endl;
									stream << ix << " " << iy << " " << iz << std::endl;
									displayOnMATLAB(stream);
									return;
								}
							}
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
		std::ofstream ofs("./out.ppm", std::ios::out | std::ios::binary); //DEBUG IMAGE
		ofs << "P6\n" << options.imageWidth << " " << options.imageHeight << "\n255\n";
		for (uint32_t j = 0; j < options.imageHeight; ++j) {
			for (uint32_t i = 0; i < options.imageWidth; ++i) {
				unsigned char r = (unsigned char)(viewOutput[i][j][0]);
				unsigned char g = (unsigned char)(viewOutput[i][j][1]);
				unsigned char b = (unsigned char)(viewOutput[i][j][2]);
				ofs << r << g << b;
			}
		}

		ofs.close();
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