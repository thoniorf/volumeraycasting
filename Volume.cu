/* Volume
 * Volumetric data
*/
#ifndef VOLUME
#define VOLUME
#endif // !VOLUME
#define _USE_MATH_DEFINES
//#define M_PI 3.14159265359
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <string>
#include <vector>
#include <ctime>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <exception>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

std::time_t timer;
int timeout = 300;

struct Options {
public:
	double threshold = 0;
	double minIntensity = 0;
	double maxIntensity = 0;
	double scale = 1;
	double viewOffset = 1;
	size_t imageWidth = 2;
	size_t imageHeight = 2;
	size_t imageDimension = 4;
	double fov = 60;
	double imageAspectRatio = 1;
	double visibleObjectsSize = 0;

	void setImageSize(double width, double height) {
		imageWidth = std::floor(width);
		imageHeight = std::floor(height);
	}

	void setIntensity(double min, double max) {
		minIntensity = min;
		maxIntensity = max;
	}

};
__host__ __device__
inline
double degToRad(const double& d) {
	return d * M_PI / 180;
}
__host__ __device__
inline
double fixZeroDoublePrecisionError(const double& x) {
	if (std::fabs(x) <= 5.00e-5)
		return 0;
	return x;
}
/* 3D VECTOR CLASS */
class Vector3 {
public:
	double x, y, z;
	__host__ __device__
	Vector3() { x = y = z = 0; }
	__host__ __device__
	Vector3(double scalar) : x(scalar), y(scalar), z(scalar) {}
	__host__ __device__
	Vector3(double i, double j, double k) : x(fixZeroDoublePrecisionError(i)), y(fixZeroDoublePrecisionError(j)), z(fixZeroDoublePrecisionError(k)) {}
	__host__ __device__
	const double& operator [] (size_t i) const { return (&x)[i]; }
	__host__ __device__
	double& operator [] (size_t i) { return (&x)[i]; }
	__host__ __device__
	Vector3 operator - () { return Vector3(-x, -y, -z); }
	__host__ __device__
	Vector3 operator + (Vector3 v) { return Vector3(x + v.x, y + v.y, z + v.z); }
	__host__ __device__
	Vector3 operator - (Vector3 v) { return Vector3(x - v.x, y - v.y, z - v.z); }
	__host__ __device__
	Vector3 operator * (Vector3 v) { return Vector3(x * v.x, y * v.y, z * v.z); }
	__host__ __device__
	Vector3 operator * (double scalar) { return Vector3(x * scalar, y * scalar, z * scalar); }
	__host__ __device__
	Vector3 operator / (double scalar) { return Vector3(x / scalar, y / scalar, z / scalar); }
	__host__ __device__
	Vector3 operator = (Vector3 v) {
		x = fixZeroDoublePrecisionError(v.x);
		y = fixZeroDoublePrecisionError(v.y);
		z = fixZeroDoublePrecisionError(v.z);
		return Vector3(x, y, z);
	}
	__host__ __device__
	bool operator != (Vector3 v) {
		return (v.x != x || v.y != y || v.z != z);
	}
	__host__ __device__
	inline
		double norm() const { return x * x + y * y + z * z; }
	__host__ __device__	
	inline
		double length() const { return sqrt(norm()); }
	__host__ __device__
	inline
		Vector3 normalize() {
		double n = norm();
		double xx = x, yy = y, zz = z;
		if (n > 0) {
			double factor = 1 / sqrt(n);
			xx *= factor, yy *= factor, zz *= factor;
		}
		return Vector3(xx, yy, zz);
	}
	__host__ __device__
	inline
		double dot(const Vector3& v) {
		return x * v.x + y * v.y + z * v.z;
	}
	__host__ __device__
	inline
		Vector3 cross(const Vector3& v) {
		double cx, cy, cz;
		cx = y * v.z - z * v.y;
		cy = z * v.x - x * v.z;
		cz = x * v.y - y * v.x;
		return Vector3(cx, cy, cz);
	}
	__host__ __device__
	friend Vector3 operator / (double scalar, Vector3 vec) {
		return Vector3(scalar / vec.x, scalar / vec.y, scalar / vec.z);
	}
	__host__ 
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
	__host__ __device__
	Matrix4x4() {}
	__host__ __device__
	Matrix4x4(Vector3 translation) {
		// translation component
		x[0][3] = translation.x;
		x[1][3] = translation.y;
		x[2][3] = translation.z;

	}
	__host__ __device__
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
	__host__ __device__
	const double* operator [] (uint8_t i) const { return x[i]; }
	__host__ __device__
	double* operator [] (uint8_t i) { return x[i]; }
	__host__ __device__
	inline
		Matrix4x4 matrixMulti(const Matrix4x4& m) {
		Matrix4x4 c;
		for (uint8_t i = 0; i < 4; ++i) {
			for (uint8_t j = 0; j < 4; ++j) {
				c[i][j] = x[i][0] * m[0][j] + x[i][1] * m[1][j] + x[i][2] * m[2][j] + x[i][3] * m[3][j];
			}
		}
		return c;
	}
	__host__ __device__
	inline
		Vector3 VecMatrixMulti(const Vector3& src) {
		double a, b, c, w;

		a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0] + x[3][0];
		b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1] + x[3][1];
		c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2] + x[3][2];
		w = src.x * x[0][3] + src.y * x[1][3] + src.z * x[2][3] + x[3][3];
		return Vector3(a / w, b / w, c / w);
	}
	__host__ __device__
	inline
		Vector3 DirMatrixMulti(const Vector3& src) const
	{
		double a, b, c;

		a = src.x * x[0][0] + src.y * x[1][0] + src.z * x[2][0];
		b = src.x * x[0][1] + src.y * x[1][1] + src.z * x[2][1];
		c = src.x * x[0][2] + src.y * x[1][2] + src.z * x[2][2];

		return Vector3(a, b, c);
	}
	__host__ __device__
	inline
		Vector3 ColMajMatrixMulti(const Vector3& src) const
	{
		double a, b, c;

		a = src.x * x[0][0] + src.y * x[0][1] + src.z * x[0][2];
		b = src.x * x[1][0] + src.y * x[1][1] + src.z * x[1][2];
		c = src.x * x[2][0] + src.y * x[2][1] + src.z * x[2][2];

		return Vector3(a, b, c);
	}
	__host__
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

class Color {
protected:
	Vector3 arancione;
	Vector3 rosso;
	Vector3 giallo;
	Vector3 fucsia;
	Vector3 verdino;
	Vector3 rosa;
	Vector3 azzurro;
	Vector3 default;
public:
	__host__
	Color() {
		arancione = Vector3(1, 0.65, 0);
		rosso = Vector3(1, 0, 0);
		giallo = Vector3(1, 1, 0);
		fucsia = Vector3(0.65, 0.12, 0.94);
		verdino = Vector3(0.48, 0.99, 0);
		rosa = Vector3(1, 0.75, 0.80);
		azzurro = Vector3(0.68, 0.85, 0.90);
		default = Vector3(0.5, 0.5, 0.5);
	}
	__device__
	Vector3 getColorByIndex(size_t index) {
		switch (index)
		{
		case 0: return arancione;
		case 1: return rosso;
		case 2: return giallo;
		case 3: return fucsia;
		case 4: return verdino;
		case 5: return rosa;
		case 6: return azzurro;
		default:
			return default;
		}
	}
};
class YuvColor : public Color {
public:
	__host__
	YuvColor() {
		arancione = Vector3(1, -0.318368, 0.251903);
		rosso = Vector3(1, -0.09991, 0.615);
		giallo = Vector3(1, -0.436, 0.05639);
		fucsia = Vector3(1, 0.304568, 0.27971);
		verdino = Vector3(1, -0.380686, -0.257824);
		rosa = Vector3(1, -0.0031775, 0.15093);
		azzurro = Vector3(1, 0.0387847, -0.10737);
		default = Vector3(1, 0, 0);
	}
};

/* AABBOX CLASS */
class Grid {
public:
	Vector3 bounds[2] = { 0 };
	__host__ __device__
	Grid() {}
	__host__ __device__
	Grid(const Vector3 &min, const Vector3 &max) {
		bounds[0] = min, bounds[1] = max;
	}
	__device__
	bool isOutsideGrid(int ix, int iy, int iz) {
		return ix < 0 || iy < 0 || iz < 0 || ix > bounds[1].x - 1 || iy > bounds[1].y - 1 || iz > bounds[1].z - 1;
	}
	__device__
	inline
		bool isInsideGrid(const int& i, const int& j, const int& k) {
		return i > 0 && j > 0 && k > 0 && i < bounds[1].x - 1 && j < bounds[1].y - 1 && k < bounds[1].z - 1;
	}
	__host__
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
	__device__
	Ray(const Vector3 &startPoint, const Vector3 &endPoint) :orig(startPoint), end(endPoint) {
		length = end - orig;
		dir = length.normalize();
		invDir = 1 / dir;
	}
	__device__
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
	double theta, phi, sigma;

	Camera() {}

	Camera(Vector3 center) {
		this->center = center;
		worldTransformation = Matrix4x4(center);

	}

	Camera(Vector3 from, Vector3 to) {
		this->from = from;
		this->to = to;
		this->rotationM = Matrix4x4();
		/*setupLookAtMatrix();*/
	}

	void yaw(const double& angle) { // z-axis
		theta = degToRad(std::fmod(angle, 360));
		Matrix4x4 rZ = Matrix4x4(std::cos(theta), -1 * std::sin(theta), 0, 0,
			std::sin(theta), std::cos(theta), 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1);
		rotationM = rotationM.matrixMulti(rZ);
	}

	void pitch(const double& angle) { //y-axis
		phi = degToRad(std::fmod(angle, 360));
		Matrix4x4 rY = Matrix4x4(std::cos(phi), 0, -1 * std::sin(phi), 0,
			0, 1, 0, 0,
			std::sin(phi), 0, std::cos(phi), 0,
			0, 0, 0, 1);
		rotationM = rotationM.matrixMulti(rY);
	}

	void roll(const double& angle) {
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
		Vector3 tmp = rotationM.DirMatrixMulti(Vector3(0, 1, 0));
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
__device__
inline
bool computeRayABBoxIntersection(const Ray& ray, double& tmin, double& tmax, const Grid& grid) {
	double  tminy, tmaxy, tminz, tmaxz;
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

	if (tmin > tmaxy || tminy > tmax) return false;

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

	if (tmin > tmaxz || tminz > tmax) return false;

	if (tminz > tmin) tmin = tminz;
	if (tmaxz < tmax) tmax = tmaxz;

	return true;
}
__device__
double interpolation(double x, Options option) {
	if (x < option.threshold) return option.minIntensity;
	return  (x - option.minIntensity) * ((255) / (option.maxIntensity - option.minIntensity));
}
__device__
double interpolation01(double x, Options option) {
	return  (x - option.minIntensity) * ((1) / (option.maxIntensity - option.minIntensity));
}
__device__
Vector3 rasterToScreen(size_t w, size_t h, Options options) {
	// from raster to normalized to normalized to screen to camera
	double cameraX = (2 * (w + 0.5) / options.imageWidth - 1) * options.scale * options.imageAspectRatio;
	double cameraY = (1 - 2 * (h + 0.5) / options.imageHeight) * options.scale;

	return Vector3(cameraX, cameraY, -1);
}
__device__
Vector3 rasterToScreen(size_t w, size_t h, double z, Options options) {
	// from raster to normalized to normalized to screen to camera
	double cameraX = (2 * (w + 0.5) / options.imageWidth - 1) * options.imageAspectRatio;//*options.scale;
	double cameraY = (1 - 2 * (h + 0.5) / options.imageHeight);//*options.scale;

	return Vector3(cameraX, cameraY, z);
}
__device__
inline
Vector3 getGradient(const int& ix, const int& iy, const int& iz, const mxDouble * volume, const mwSize* size) {
	// component-wise linear interpolation
	double a, b, c;
	int xi = (int)(ix + 0.5);
	double xT = ix + 0.5 - xi;
	int linearXi = xi + iy * size[0] + iz * (size[0] * size[1]);
	int linearXiB = xi+1 + iy * size[0] + iz * (size[0] * size[1]);
	int linearXib = xi-1 + iy * size[0] + iz * (size[0] * size[1]);

	a = (volume[linearXi] - volume[linearXib]) * (1.0 - xT) + (volume[linearXiB] - volume[linearXi]) * xT;
	
	int yi = (int)(iy + 0.5);
	double yT = iy + 0.5 - yi;
	int linearYi = ix + yi * size[0] + iz * (size[0] * size[1]);
	int linearYiB = ix + (yi+1) * size[0] + iz * (size[0] * size[1]);
	int linearYib = ix + (yi-1) * size[0] + iz * (size[0] * size[1]);
	
	b = (volume[linearYi] - volume[linearYib]) * (1.0 - yT) + (volume[linearYiB] - volume[linearYi]) * yT;
	
	int zi = (int)(iz + 0.5);
	double zT = iz + 0.5 - zi;
	int linearZi = ix + iy * size[0] + zi * (size[0] * size[1]);
	int linearZiB = ix + iy * size[0] + (zi+1) * (size[0] * size[1]);
	int linearZib = ix + iy * size[0] + (zi-1) * (size[0] * size[1]);
	
	c = (volume[linearZi] - volume[linearZib]) * (1.0 - zT) + (volume[linearZiB] - volume[linearZi]) * zT;
	
	return Vector3(a, b, c);
}

__global__
void raycasting(mxDouble * viewOutput, mxDouble const * const volume, mxDouble const * const objectVolume,
				mwSize const * const size, int const * const visibleObj,double const *const visibleAlpha,
				Options options, Matrix4x4 lookAt, Vector3 from, 
				Vector3 lightPosition, Vector3 ambientColor,Vector3 specularColor, double shininess, double specularity,
				Color colors, YuvColor yuvColors, Matrix4x4 rgbToYuv, Matrix4x4 yuvToRgb,
				Grid grid, size_t maxStep) {
	Vector3 diffuseColor(0.5, 0.5, 0.5);
	double littleStep = 1;
	double tmin, tmax;
	int const i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < options.imageDimension) {
		int x = (i % options.imageWidth)-options.imageWidth/2;
		int y = (i / options.imageWidth)-options.imageHeight/2;
		for (size_t iObj = 0; iObj < options.visibleObjectsSize; ++iObj) {

			Vector3 rayStart = lookAt.VecMatrixMulti(Vector3(x, y, 0));
			Vector3 rayEnd = lookAt.VecMatrixMulti(Vector3(x, y, 1));

			Ray ray(rayStart, rayEnd);

			if (computeRayABBoxIntersection(ray, tmin, tmax, grid)) {

				Vector3 start = ray.orig + ray.dir *tmin;
				Vector3 end = ray.orig + ray.dir *tmax;

				for (size_t t = 0; t < maxStep; t += littleStep) {
					int ix = floor(start.x) > 0 ? floor(start.x) - 1 : 0;
					int iy = floor(start.y) > 0 ? floor(start.y) - 1 : 0;
					int iz = floor(start.z) > 0 ? floor(start.z) - 1 : 0;
					int linearIndex = ix + iy * size[0] + iz * (size[0] * size[1]);
					if (grid.isInsideGrid(ix, iy, iz)) {
						if (volume[linearIndex] >= options.threshold && (visibleObj[iObj] == -1|| objectVolume[linearIndex] == visibleObj[iObj])) {

							Vector3 grad = getGradient(ix, iy, iz, volume,size);
							Vector3 normal = -grad / std::sqrt(grad.norm());

							Vector3 lightDir = from.normalize();
							double distance = lightPosition.length();
							double lambertian = lightDir.dot(normal);

							Vector3 viewDir = rayStart.normalize();
							Vector3 halfDir = (lightDir + viewDir).normalize();

							double specAngle = halfDir.dot(normal);
							double specular = std::pow(specAngle, shininess);

							diffuseColor = colors.getColorByIndex(visibleObj[iObj]);

							// 1* lightAmbientColor  +  1* lightDiffuseColor*dot(lightdir,normals) * weight  +  1* lightSpecularColor * [dot(halfdir,normals)]^shininess * (1-weight)
							Vector3 IlluminationI = ambientColor + diffuseColor * lambertian * specularity + specularColor * specular * (1 - specularity);
							//convert to yuv
							Vector3 yuvIllumination = rgbToYuv.ColMajMatrixMulti(IlluminationI);
							//get diffuse color in yuv color schema
							Vector3 yuvDiffuse = yuvColors.getColorByIndex(visibleObj[iObj]);
							//reset color
							yuvIllumination.y = yuvDiffuse.y;
							yuvIllumination.z = yuvDiffuse.z;
							//convert back to rgb
							IlluminationI = yuvToRgb.ColMajMatrixMulti(yuvIllumination);

							// clamp new illumination to valid rgb value.ie: x<0-> x=0, x>1 -> x=1
							viewOutput[i] += visibleAlpha[iObj] * IlluminationI.x;
							viewOutput[i + options.imageDimension] += visibleAlpha[iObj] * IlluminationI.y;
							viewOutput[i + options.imageDimension * 2] += visibleAlpha[iObj] * IlluminationI.z;

							/*viewOutput[i] = 0.7;
							viewOutput[i + options.imageDimension] = 0.5;
							viewOutput[i + options.imageDimension * 2] = 0.3;*/
							break;
						}
					}
					start = start + ray.dir*littleStep;

				}

			}
		}
	}
	return;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	std::time(&timer);
	Options options;
	/*
	* GPU INIT
	* CREATING OR REFERINCING VARIABLE IN DEVICE MEMORY
	*/
	mxInitGPU();

	mxGPUArray const * d_VolumeArray = mxGPUCreateFromMxArray(prhs[0]);
	const mxDouble *  d_Volume = (mxDouble const*)mxGPUGetDataReadOnly(d_VolumeArray);
	const mwSize *sizeArray = mxGetDimensions(prhs[0]);
	const mwSize numberOfVoxels = mxGetNumberOfElements(prhs[0]);
	mwSize * d_Size;
	cudaMalloc((void**)&d_Size, 3 * sizeof(mwSize));
	cudaMemcpy(d_Size, sizeArray, 3 * sizeof(mwSize), cudaMemcpyHostToDevice);

	mxGPUArray const * d_ObjectsArray = mxGPUCreateFromMxArray(prhs[1]);
	const mxDouble *  d_ObjectVolume = (mxDouble const*)mxGPUGetDataReadOnly(d_ObjectsArray);
	const mwSize *objectSizeArray = mxGetDimensions(prhs[0]);
	const mwSize numberOfObjects = mxGetNumberOfElements(prhs[1]);

	const mxDouble *thresholdArray = mxGetPr(mxGetField(prhs[2], 0, "threshold"));
	const mxDouble *viewArray = mxGetPr(mxGetField(prhs[2], 0, "view"));
	const mxDouble *alphaArray = mxGetPr(mxGetField(prhs[2], 0, "alpha"));
	const mxDouble *specularityArray = mxGetPr(mxGetField(prhs[2], 0, "specularity"));
	const mxDouble *intensityArray = mxGetPr(mxGetField(prhs[2], 0, "intensity"));
	const mxDouble *rotationArray = mxGetPr(mxGetField(prhs[2], 0, "rotation"));
	const mxDouble *colorArray = mxGetPr(mxGetField(prhs[2], 0, "colors"));

	const mwSize alphaSize = mxGetNumberOfElements(mxGetField(prhs[2], 0, "alpha"));

	options.setImageSize(viewArray[0], viewArray[1]);
	options.imageDimension = options.imageHeight * options.imageWidth;
	options.setIntensity(intensityArray[0], intensityArray[1]);
	options.threshold = (thresholdArray[0]);
	options.fov = 52.51;
	options.viewOffset = 200;
	options.scale = std::tan(degToRad((options.fov * 0.5)));
	options.imageAspectRatio = options.imageWidth / options.imageHeight;

	mwSize frameDimensions[3] = { options.imageWidth,options.imageHeight,3 };

	mxGPUArray * const  d_viewArray = mxGPUCreateGPUArray(3, frameDimensions, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
	mxDouble * d_viewOutput = (mxDouble *)mxGPUGetData(d_viewArray);
#if DEBUG
	std::cout << "SIZE " << sizeArray[0] << " " << sizeArray[1] << " " << sizeArray[2] << std::endl
		<< "VOX " << numberOfVoxels << std::endl
		<< "VIEW " << viewArray[0] << " " << viewArray[1] << std::endl
		<< "ROT " << rotationArray[0] << " " << rotationArray[1] << " " << rotationArray[2] << std::endl
		<< "INT " << intensityArray[0] << " " << intensityArray[1] << std::endl
		<< "THR " << thresholdArray[0] << std::endl
		<< "OBJS SIZE " << objectSizeArray[0] << " " << objectSizeArray[1] << " " << objectSizeArray[2] << std::endl;
#endif

	if (numberOfObjects > 0) {
		for (int j = 0; j < alphaSize; ++j) {
			if (alphaArray[j] > 0.0) {
				options.visibleObjectsSize++;
			}
		}
	}
	else {
		options.visibleObjectsSize = 1;
	}
	
	int * visibleObj = new int[options.visibleObjectsSize];
	double * visibleAlpha = new double[options.visibleObjectsSize];

	if (numberOfObjects == 0) {
		visibleAlpha[0] = 1;
		visibleObj[0] = -1;
	}
	else {
		int visObj = 0;
		for (size_t i = 0; i < alphaSize; ++i) {
			if (alphaArray[i] > 0) {
				visibleAlpha[visObj] = alphaArray[i];
				visibleObj[visObj] = i;
				visObj++;
			}
		}
	}

	Matrix4x4 rgbToYuv(0.2126, 0.7152, 0.0722, 0, -0.09991, -0.33609, 0.436, 0, 0.615, -0.55861, -0.05639, 0, 0, 0, 0, 0);
	Matrix4x4 yuvToRgb(1, 0, 1.28033, 0, 1, -0.21482, -0.38059, 0, 1, 2.12798, 0, 0, 0, 0, 0, 0);

	Vector3 lightPosition(sizeArray[0] / 2 + options.viewOffset, sizeArray[1] / 2, sizeArray[2] / 2);

	Vector3 ambientColor(0.5, 0.5, 0.5);
	Vector3 diffuseColor(0.6, 0.6, 0.6);
	Vector3 specularColor(0.7, 0.7, 0.7);

	double shininess = 16.0;
	double specularity = specularityArray[0];

	Grid grid(Vector3(0), Vector3(sizeArray[0], sizeArray[1], sizeArray[2]));

	Camera camera(Vector3(sizeArray[0] / 2 + options.viewOffset, sizeArray[1] / 2, sizeArray[2] / 2), Vector3(sizeArray[0] / 2, sizeArray[1] / 2, sizeArray[2] / 2));

	camera.pitch(rotationArray[1]);
	camera.roll(rotationArray[2]);
	camera.yaw(rotationArray[0]);
	camera.setupLookAtMatrix();

	size_t maxStep = std::sqrt(sizeArray[0] * sizeArray[0] + sizeArray[1] * sizeArray[1] + sizeArray[2] * sizeArray[2]);

	double halfWidth = std::fabs(options.imageWidth / 2);
	double halfHeight = std::fabs(options.imageHeight / 2);

	Color colors;
	YuvColor yuvColors;

	cudaError_t err = cudaSuccess;
	int* d_visibleObj;
	cudaMalloc((void**)&d_visibleObj, options.visibleObjectsSize * sizeof(int));
	err = cudaMemcpy(d_visibleObj, visibleObj, options.visibleObjectsSize * sizeof(int), cudaMemcpyHostToDevice);
	double* d_visibleAlpha;
	cudaMalloc((void**)&d_visibleAlpha, options.visibleObjectsSize * sizeof(double));
	err = cudaMemcpy(d_visibleAlpha, visibleAlpha, options.visibleObjectsSize * sizeof(double), cudaMemcpyHostToDevice);

	//call kernel
	raycasting<<<std::ceil(options.imageDimension / 256.0), 256 >>>(d_viewOutput, d_Volume, d_ObjectVolume, d_Size, d_visibleObj,d_visibleAlpha,
																	options, camera.lookAt, camera.from,
																	lightPosition,ambientColor,specularColor,shininess,specularity,
																	colors,yuvColors,rgbToYuv,yuvToRgb,
																	grid, maxStep);

	// uncomment to show GPU work
	plhs[0] = mxGPUCreateMxArrayOnGPU(d_viewArray);
	delete [] visibleObj;
	delete [] visibleAlpha;
	cudaFree(d_visibleAlpha);
	cudaFree(d_visibleObj);
	cudaFree(d_Size);
	mxGPUDestroyGPUArray(d_viewArray);
	mxGPUDestroyGPUArray(d_ObjectsArray);
	mxGPUDestroyGPUArray(d_VolumeArray);
	return;
}