// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Created by L.Zampetti (C) 2017 
// ROW NUMBERS -87

#extension GL_OES_standard_derivatives : enable

#ifdef GL_ES
	precision mediump float;
#endif

vec2 u_resolution = iResolution.xy;
vec2 u_mouse = iMouse.xy;
float u_time = iGlobalTime;
// sampler2D u_texture = iChannel0;

/*
#ifdef iResolution
    #define u_resolution    iResolution
    #define u_mouse         iMouse
    #define u_time          iGlobalTime
    #define u_texture       iChannel0
#else
    uniform vec2 u_resolution;
    uniform vec2 u_mouse;
    uniform float u_time;
    uniform sampler2D u_texture;
#endif
*/

#define PI						3.14159265359
#define TWO_PI          		6.28318530718
#define RAD						0.01745329251
#define EPSILON					0.001

// GLOBALS
vec2 st;
vec2 uv;
vec2 vp;
vec2 mx;
vec2 pixel;
vec4 point;
mat4 matrix;

// COLORS
vec3 white = 	vec3(1.0, 1.0, 1.0);
vec3 red = 		vec3(1.0, 0.0, 0.0);
vec3 green = 	vec3(0.0, 1.0, 0.0);
vec3 blue = 	vec3(0.0, 0.0, 1.0);
vec3 yellow = 	vec3(1.0, 1.0, 0.0);
vec3 magenta = 	vec3(1.0, 0.0, 1.0);
vec3 cyan = 	vec3(0.0, 1.0, 1.0);

// MATH
const mat4 projection = mat4(
	vec4(3.0 / 4.0, 0.0, 0.0, 0.0),
	vec4(     0.0, 1.0, 0.0, 0.0),
	vec4(     0.0, 0.0, 0.5, 0.5),
	vec4(     0.0, 0.0, 0.0, 1.0)
);
mat4 scale = mat4(
	vec4(4.0 / 3.0, 0.0, 0.0, 0.0),
	vec4(     0.0, 1.0, 0.0, 0.0),
	vec4(     0.0, 0.0, 1.0, 0.0),
	vec4(     0.0, 0.0, 0.0, 1.0)
);
mat4 rotation = mat4(
	vec4(1.0,          0.0,         0.0, 	0.0),
	vec4(0.0,  cos(u_time), sin(u_time),  	0.0),
	vec4(0.0, -sin(u_time), cos(u_time),  	0.0),
	vec4(0.0,          0.0,         0.0, 	1.0)
);
mat4 rotationAxis(float angle, vec3 axis) {
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0 - c;
    return mat4(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,  0.0,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,  0.0,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c,           0.0,
                0.0,                                0.0,                                0.0,                                1.0);
}
vec2 toScreen(vec4 p) {
	vec4 screen = projection * scale * p;
	float perspective = screen.z * 0.5 + 1.0;
    screen /= perspective;
	return screen.xy;
}
vec3 rotateX(vec3 p, float angle) {
	mat4 rmy = rotationAxis(angle, vec3(1.0, 0.0, 0.0));
	return (vec4(p, 1.0) * rmy).xyz;
}
vec3 rotateY__(vec3 p, float angle) {
	mat4 rmy = rotationAxis(angle, vec3(0.0, 1.0, 0.0));
	return (vec4(p, 1.0) * rmy).xyz;
}
vec3 rotateZ(vec3 p, float angle) {
	mat4 rmy = rotationAxis(angle, vec3(0.0, 0.0, 1.0));
	return (vec4(p, 1.0) * rmy).xyz;
}
vec3 rotateY(vec3 p, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    mat4 r = mat4(
        vec4(c, 0, s, 0),
        vec4(0, 1, 0, 0),
        vec4(-s, 0, c, 0),
        vec4(0, 0, 0, 1)
    );
	return (vec4(p, 1.0) * r).xyz;
}

// VARS
vec2 getSt() {
	vec2 st = gl_FragCoord.xy / u_resolution.xy;
	st.y *= u_resolution.y / u_resolution.x;
	st.y += (u_resolution.x - u_resolution.y) / u_resolution.x / 2.0;
	return st;
}
vec2 getUv(vec2 st) {
	vec2 uv = -1.0 + st * 2.0;
	return uv;
}
vec2 getMx() {
	return -1.0 + u_mouse / u_resolution.xy * 2.0;
}
vec4 getNormal() {
	return vec4(normalize( cross( dFdx( point.xyz ), dFdy( point.xyz ) ) ), 1.0);
}
void setVars() {
	st = getSt();
	uv = getUv(st);
	mx = getMx();
	if (false) {
		point = vec4(uv, 0.0, 1.0);
		matrix = rotationAxis(
			RAD * 360.0 * u_time * 0.2 + st.x * st.y, 
			vec3(1.0, 1.0, 1.0)
		);
		// matrix = rotationAxis(RAD * 180.0 * cos(u_time), vec3(0.0, 1.0, 0.0));
		// matrix *= rotationAxis(RAD * 180.0 * sin(u_time), vec3(1.0, 0.0, 0.0));
		point *= matrix;
	} else {
		vec2 p01 = gl_FragCoord.xy / u_resolution.xy;
		float 	r = .8, 
				s = p01.y * 1.0 * TWO_PI, 
				t = p01.x * 1.0 * PI;
		point = vec4(
			r * cos(s) * sin(t),
			r * sin(s) * sin(t),
			r * cos(t),
			1.0
		);
		matrix = rotationAxis(RAD * 360.0 * u_time * .02, vec3(1.0, 1.0, 1.0));
		point *= matrix;
	}
	// pixel = vec2(1.0) / u_resolution * 100.0;
	pixel = vec2(0.003);
}

vec3 drawLine(vec4 p, float y, vec3 color, float size) {
	vec2 screen = toScreen(p);
	return color * (smoothstep(y - pixel.x * size, y, screen.y) - smoothstep(y, y + pixel.y * size, screen.y));
}
vec3 drawLine(vec4 p, float y, vec3 color) {
	return drawLine(p, y, color, 6.0);
}
vec3 drawLine(vec4 p, float y) {
	return drawLine(p, y, white, 6.0);
}
vec3 drawPoint(vec4 p, vec3 color, float size) {
	// pixel += abs(cos(u_time)) * 0.1;
	vec2 screen = toScreen(p);
	float l = length(uv - screen);
	if (l < pixel.x * size) {
		return color * smoothstep(l - pixel.x * size, l, size);
	} else {
		return vec3(0.0);
	}	
}
vec3 drawPoint(vec4 p, vec3 color) {
	return drawPoint(p, color, 2.0);	
}
vec3 drawPoint(vec4 p) {
	return drawPoint(p, white, 2.0);	
}

// GEOMETRIES
// Sphere - signed - exact
float sphere(vec3 p, float s) {
  return length(p)-s;
}
// Box - unsigned - exact
float ubox(vec3 p, vec3 b) {
  return length(max(abs(p)-b,0.0));
}
// Round Box - unsigned - exact
float roundBox(vec3 p, vec3 b, float r) {
  return length(max(abs(p)-b,0.0))-r;
}
// Box - signed - exact
float box(vec3 p, vec3 b) {
  vec3 d = abs(p) - b;
  return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}
// Torus - signed - exact
float torus(vec3 p, vec2 t) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
// Cylinder - signed - exact
float cylinder(vec3 p, vec3 c) {
  return length(p.xz-c.xy)-c.z;
}
// Cone - signed - exact
float cone(vec3 p, vec2 c) {
    // c must be normalized
    float q = length(p.xy);
    return dot(c,vec2(q,p.z));
}
// Plane - signed - exact
float plane(vec3 p, vec4 n) {
  // n must be normalized
  return dot(p,n.xyz) + n.w;
}
// Hexagonal Prism - signed - exact
float hexPrism(vec3 p, vec2 h) {
    vec3 q = abs(p);
    return max(q.z-h.y,max((q.x*0.866025+q.y*0.5),q.y)-h.x);
}
// Triangular Prism - signed - exact
float triPrism(vec3 p, vec2 h) {
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}
// Capsule / Line - signed - exact
float capsule(vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}
// Capped cylinder - signed - exact
float cappedCylinder(vec3 p, vec2 h) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - h;
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
// Capped Cone - signed - bound
float cappedCone(in vec3 p, in vec3 c) {
    vec2 q = vec2( length(p.xz), p.y );
    vec2 v = vec2( c.z*c.y/c.x, -c.z );
    vec2 w = v - q;
    vec2 vv = vec2( dot(v,v), v.x*v.x );
    vec2 qv = vec2( dot(v,w), v.x*w.x );
    vec2 d = max(qv,0.0)*qv/vv;
    return sqrt( dot(w,w) - max(d.x,d.y) ) * sign(max(q.y*v.x-q.x*v.y,w.y));
}
// Ellipsoid - signed - bound
float ellipsoid(in vec3 p, in vec3 r) {
    return (length( p/r ) - 1.0) * min(min(r.x,r.y),r.z);
}

// boolean operation
float bIntersect(float da, float db) {
    return max(da, db);
}
float bUnion(float da, float db) {
    return min(da, db);
}
float bDifference(float da, float db) {
    return max(da, -db);
}
// bend
float twistedCube(vec3 p) {
    vec3 q = rotateY(p, 0.5 * p.y);
    return box(q, vec3(0.2));
}

// NOISE
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec2 mod289(vec2 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec3 permute(vec3 x) { return mod289(((x * 34.0) + 1.0) * x); }
// simplex noise
float snoise(vec2 v) {
    const vec4 C = vec4(
		 0.211324865405187,  	// (3.0-sqrt(3.0))/6.0
		 0.366025403784439,  	// 0.5*(sqrt(3.0)-1.0)
		-0.577350269189626,  	// -1.0 + 2.0 * C.x
		 0.024390243902439 		// 1.0 / 41.0
	);
    vec2 i  = floor(v + dot(v, C.yy));
    vec2 x0 = v -   i + dot(i, C.xx);
    vec2 i1;
    i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    vec4 x12 = x0.xyxy + C.xxzz;
    x12.xy -= i1;
    i = mod289(i); // Avoid truncation effects in permutation
    vec3 p = permute(permute(i.y + vec3(0.0, i1.y, 1.0 )) + i.x + vec3(0.0, i1.x, 1.0));
    vec3 m = max(0.5 - vec3(dot(x0, x0), dot(x12.xy, x12.xy), dot(x12.zw, x12.zw)), 0.0);
    m = m * m;
    m = m * m;
    vec3 x = 2.0 * fract(p * C.www) - 1.0;
    vec3 h = abs(x) - 0.5;
    vec3 ox = floor(x + 0.5);
    vec3 a0 = x - ox;
    m *= 1.79284291400159 - 0.85373472095314 * (a0 * a0 + h * h);
    vec3 g;
    g.x  = a0.x  * x0.x  + h.x  * x0.y;
    g.yz = a0.yz * x12.xz + h.yz * x12.yw;
    return 130.0 * dot(m, g);
}
/* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
vec3 random3(vec3 c) {
	float j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
	vec3 r;
	r.z = fract(512.0*j);
	j *= .125;
	r.x = fract(512.0*j);
	j *= .125;
	r.y = fract(512.0*j);
	return r-0.5;
}

const float F3 =  0.3333333;
const float G3 =  0.1666667;
float snoise(vec3 p) {
	vec3 s = floor(p + dot(p, vec3(F3)));
	vec3 x = p - s + dot(s, vec3(G3));
	vec3 e = step(vec3(0.0), x - x.yzx);
	vec3 i1 = e*(1.0 - e.zxy);
	vec3 i2 = 1.0 - e.zxy*(1.0 - e);
	vec3 x1 = x - i1 + G3;
	vec3 x2 = x - i2 + 2.0*G3;
	vec3 x3 = x - 1.0 + 3.0*G3;
	vec4 w, d;
	w.x = dot(x, x);
	w.y = dot(x1, x1);
	w.z = dot(x2, x2);
	w.w = dot(x3, x3);
	w = max(0.6 - w, 0.0);
	d.x = dot(random3(s), x);
	d.y = dot(random3(s + i1), x1);
	d.z = dot(random3(s + i2), x2);
	d.w = dot(random3(s + 1.0), x3);
	w *= w;
	w *= w;
	d *= w;
	return dot(d, vec4(52.0));
}
//
/*
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        scene(vec3(p.x + EPSILON, p.y, p.z)) - scene(vec3(p.x - EPSILON, p.y, p.z)),
        scene(vec3(p.x, p.y + EPSILON, p.z)) - scene(vec3(p.x, p.y - EPSILON, p.z)),
        scene(vec3(p.x, p.y, p.z  + EPSILON)) - scene(vec3(p.x, p.y, p.z - EPSILON))
    ));
}
*/

vec3 rotatePoint(vec3 p) {
	return (vec4(p, 1.0) * matrix).xyz;
}

// WTF has this clown done here? Good question. A smooth spherical surface doesn't really do raymarching enough justice, so I've added some cheap bumps.
// A sphere is easy to raytrace, and raymarch for that matter. The function below can be raytraced, but is only "easy" to do in a raymarching setting.
// The term "cheap" is relative, of course. A sinusoidal function tends to be a CPU killer, whereas the GPU seems to take it in its stride.
//
// Anyway, A*sin(f*p.x)*sin(f*p.y)*sin(f*p.z) - "f" for frequency and "A" for amplitude - is just one of the ways to put bumps on a sphere. I've also
// added an extra term with half the amplitude and twice the frequency for a little more detail. Yes, smooth noise, Voronoi, etc, would be more
// interesting, but this is just a simple template. The "time" term moves the bumps around a little, which gives the object a bit of movement and gives
// the specular lighting a better chance to shine... See what I did there? I also could have gone with a corny "in the spotlight" analogy.
//
// By the way, it's actually possible to to perform some layering, combined with 3D rotations, to make this "much" nicer. In fact, possibly better than
// cheap noise, but we'll keep things simple, for now.
float sinusoidBumps(in vec3 p) {
    return  sin(p.x * 16.0 + u_time * 0.57) * 
			cos(p.y * 16.0 + u_time * 2.17) * 
			sin(p.z * 16.0 - u_time * 1.31) + 0.5 * 
			sin(p.x * 32.0 + u_time * 0.07) * 
			cos(p.y * 32.0 + u_time * 2.11) * 
			sin(p.z * 32.0 - u_time * 1.23);
}

// repetitions
float opRep(vec3 p, vec3 c) {
    vec3 q = mod(p, c) -0.5 * c;
    return sphere(q, 0.05);
	// return sphere(q, 0.05 + 0.03 * cos(u_time * 3.0 + q.y * 90.0 + q.x * 30.0));
}

// MATERIALS
struct Material {
	vec3 color;
	float ambient; // The object's ambient property. You can also have a global and light ambient property, but we'll try to keep things simple.	
	float glossiness;
	float shininess;
};
Material material;

// ###############
// ###  SCENE  ###
// ###############
#define SC (250.0)
// value noise, and its analytical derivatives
/*
vec3 noised(in vec2 x) {
    vec2 f = fract(x);
    vec2 u = f*f*(3.0-2.0*f);
#if 1
    // texel fetch version
    ivec2 p = ivec2(floor(x));
    float a = texelFetch( iChannel0, (p+ivec2(0,0))&255, 0 ).x;
	float b = texelFetch( iChannel0, (p+ivec2(1,0))&255, 0 ).x;
	float c = texelFetch( iChannel0, (p+ivec2(0,1))&255, 0 ).x;
	float d = texelFetch( iChannel0, (p+ivec2(1,1))&255, 0 ).x;
#else    
    // texture version    
    vec2 p = floor(x);
	float a = textureLod( iChannel0, (p+vec2(0.5,0.5))/256.0, 0.0 ).x;
	float b = textureLod( iChannel0, (p+vec2(1.5,0.5))/256.0, 0.0 ).x;
	float c = textureLod( iChannel0, (p+vec2(0.5,1.5))/256.0, 0.0 ).x;
	float d = textureLod( iChannel0, (p+vec2(1.5,1.5))/256.0, 0.0 ).x;
#endif   
	return vec3(a+(b-a)*u.x+(c-a)*u.y+(a-b-c+d)*u.x*u.y, 6.0*f*(1.0-f)*(vec2(b-a,c-a)+(a-b-c+d)*u.yx));
}
const mat2 m2 = mat2(0.8, -0.6, 0.6, 0.8);
float terrainM(in vec2 x) {
	vec2  p = x * 0.003 / SC;
    float a = 0.0;
    float b = 1.0;
	vec2  d = vec2(0.0);
    for(int i = 0; i < 9; i++) {
        vec3 n = noised(p);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
		b *= 0.5;
        p = m2 * p * 2.0;
    }
	return SC * 120.0 * a;
}
*/
float noise(vec3 x) {
    vec3 p = floor(x);
    vec3 f = fract(x);
    f = f*f*(3.-2.*f);
    float n = p.x + p.y*157. + 113.*p.z;   
    vec4 v1 = fract(753.5453123*sin(n + vec4(0., 1., 157., 158.)));
    vec4 v2 = fract(753.5453123*sin(n + vec4(113., 114., 270., 271.)));
    vec4 v3 = mix(v1, v2, f.z);
    vec2 v4 = mix(v3.xy, v3.zw, f.y);
    return mix(v4.x, v4.y, f.x);
}

float rocky(vec3 p) {
	// random rotation reduces artifacts
	mat3 M = mat3(
		 0.28862355854826727, 0.6997227302779844,  0.6535170557707412,
		 0.06997493955670424, 0.6653237235314099, -0.7432683571499161,
		-0.9548821651308448,  0.26025457467376617, 0.14306504491456504
	);
	vec3 p1 = M * (p + cos(u_time * 0.05));
	vec3 p2 = M * (p1 + sin(u_time * 0.05));
	float n1 = noise(p1 * 8.);
	float n2 = noise(p2 * 16.);
	float n3 = noise(p2 * 32.);
	float n4 = noise(p1 * 128.);
	float rocky = 	0.1 * n1 * n1 + 
					0.05 * n2 * n2 + 
					0.02 * n3 * n3 + 
					0.01 * n4;
	return rocky * 0.3; // (0.2 + 0.3 * abs(cos(u_time * 0.5)));
}
float getRock(vec3 p) {
	float d = sphere(p, 0.7);
	// float d = plane(p, vec4(1.0, 0.0, 0.0, 1.0));
	return d + (d < 0.01 ? rocky(p) * 1.0 : 0.0);
}
float terrain(vec3 p, vec4 n) {
	// n must be normalized
  return dot(p,n.xyz) + n.w;
	// f(x,z) = sin(x)�sin(z)
}
float getSpheres(vec3 p) {
	return opRep(p, vec3(0.5, 3.0, 0.5));
}
float getTorus(vec3 p) {
	p = rotatePoint(p);
	return torus(p, vec2(0.5, 0.02));
}
float getClouds(vec3 p) {
	/*
	return p.y - terrainM(p.xz);
	float w = 	snoise((p.xy + u_time * 0.123) * 2.345) * 
				snoise((p.yz + u_time * 0.234) * 1.234) + 3.0;
	*/
	// float w = sinusoidBumps(p * .2) * .5 + .5;
	vec3 q = rotatePoint(p);
	return getRock(q);
	/*
	float x = fract(p.x * 3.1234);
	float y = fract(p.y * 6.1234);
	float z = fract(p.z * 9.1234);
	float w = .3 
			* (sin(u_time) + sin(y))
			* (cos(u_time) + cos(x)) 
			* (sin(u_time) + sin(z));
	return terrain(p, vec4(0.0, 1.0, 0.0, w));
	// return snoise(p.xy * 3.0) * snoise(p.yz * 3.0);
	return   0.5333333 * snoise(p)
			+0.2666667 * snoise(2.0 * p)
			+0.1333333 * snoise(4.0 * p)
			+0.0666667 * snoise(8.0 * p);
	float c = roundBox(p, vec3(0.13), 0.03); // capsule(p, vec3(0.0, 0.0, -1.0), vec3(0.0, 0.0, 1.0), 0.02);
	float s = sphere(p, .2);
	return max(c, -s);
	*/
}
float scene(vec3 p) {
	float 	a = 1000000.0, // getSpheres(p),
			b = getClouds(p),
			c = 1000000.0; // getTorus(p);
	float distance = min(a, min(b, c));
	if (distance == a) {
		material = Material(
			vec3(0.0, 0.0, 1.0), // color
			0.3, // ambient,
			0.6, // glossiness
			1.9 // shininess
		);
	} else if (distance == b) {
		material = Material(
			vec3(1.0, 1.0, 1.0), // color
			0.05, // ambient,
			0.99, // glossiness
			0.01 // shininess
		);
	} else if (distance == c) {
		material = Material(
			vec3(0.0, 1.0, 0.0), // color
			0.3, // ambient,
			0.6, // glossiness
			1.9 // shininess
		);
	}
	return distance;
}
// ###############
// ###  SCENE  ###
// ###############


// A clever way to obtain the surface normal without the need to perform difficult, and often expensive, differential calculations, etc.
// Use the surface point (p) and epsilon value (eps) to obtain the gradient along each of the individual axes (f(p.x+eps)-f(p.x-eps), etc).
// That should give you a scaled representation of (df/dx, df/dy, df/dz), which can simply be normalized to get the unit normal.
// I guess it's the 3D equivalent of obtaining the gradient of a texel in a 2D texture for bump mapping. Back in the days, when no one would tell
// you anything, and you had to figure it out for yourself. I prefer to use "f(p.x + eps) - f(p.x)" to save on cycles, but it's not quite as accurate.
#define N_TYPE	3
vec3 getNormal(in vec3 p) {
#if N_TYPE == 1
	// 6-tap normalization. Probably the most accurate, but a bit of a cycle waster.
	return normalize(vec3(
		scene(vec3(p.x + EPSILON, p.y, p.z)) - scene(vec3(p.x - EPSILON, p.y, p.z)),
		scene(vec3(p.x, p.y + EPSILON, p.z)) - scene(vec3(p.x, p.y - EPSILON, p.z)),
		scene(vec3(p.x, p.y, p.z + EPSILON)) - scene(vec3(p.x, p.y, p.z - EPSILON))
	));
#elif  N_TYPE == 2
	// Shorthand version of the above. The fewer characters used almost gives the impression that it involves fewer calculations. Almost.
	vec2 e = vec2(EPSILON, 0.0);
	return normalize(vec3(scene(p + e.xyy) - scene(p - e.xyy), scene(p + e.yxy) - scene(p - e.yxy), scene(p + e.yyx) - scene(p - e.yyx)));
#elif  N_TYPE == 3
    // If speed is an issue, here's a slightly-less-accurate, 4-tap version. If fact, visually speaking, it's virtually the same, so on a
    // lot of occasions, this is the one I'll use. However, if speed is really an issue, you could take away the "normalization" step, then
    // divide by "EPSILON," but I'll usually avoid doing that.
    float ref = scene(p);
	return normalize(vec3(
		scene(vec3(p.x + EPSILON, p.y, p.z)) - ref,
		scene(vec3(p.x, p.y + EPSILON, p.z)) - ref,
		scene(vec3(p.x, p.y, p.z + EPSILON)) - ref
	));
#elif  N_TYPE == 4
	// The tetrahedral version, which does involve fewer calculations, but doesn't seem as accurate on some surfaces... I could be wrong,
	// but that's the impression I get.
	vec2 e = vec2(-0.5 * EPSILON, 0.5 * EPSILON);
	return normalize(e.yxx * scene(p + e.yxx) + e.xxy * scene(p + e.xxy) + e.xyx * scene(p + e.xyx) + e.yyy * scene(p + e.yyy));
#endif
}

// CAMERA
struct Camera {
    vec3 position;
    vec3 target;
    vec3 forward;
    vec3 right;
    vec3 up;
    float fov;
	float near;
	float far;
};
Camera getCamera(vec3 position, vec3 target) {
	// position. This is the point you look from, or camera you look at the scene through. Whichever way you wish to look at it.
	// target. This is the point you look towards, or at.
    // forward. Forward vector.
	// right. Right vector... or is it left? Either way, so long as the correct-facing up-vector is produced.
    // up. Cross product the two vectors above to get the up vector.
    // fov. Field of view. Make it bigger, and the screen covers a larger area, which means more of the scene can be seen. This, in turn, means that our objects will appear smaller.
	const float fov = 0.25;
	const float near = 0.0;
	const float far = 200.0;
	Camera camera = Camera(position, target, normalize(target - position), vec3(0.0), vec3(0.0), fov, near, far);
	camera.right = normalize(vec3(camera.forward.z, 0.0, -camera.forward.x));
	camera.up = normalize(cross(camera.forward, camera.right));
	return camera;
}

// MARCHER
const int STEPS = 48; // 128	
struct Marcher {
    vec3 origin;
    vec3 direction;
	float scale;
	float threshold;
	float distance;
	float depth;
};
Marcher getMarcher(Camera camera) {
	const float scale = 0.5;
	const float threshold = 0.005; // I'm not quite sure why, but thresholds in the order of a pixel seem to work better for me... most times.
	// origin. Ray origin. Every ray starts from this point, then is cast in the rd direction.
    // direction. Ray direction. This is our one-unit-long direction ray.
    Marcher marcher = Marcher(
		camera.position,
		normalize(camera.forward + camera.fov * uv.x * camera.right + camera.fov * uv.y * camera.up),
		scale,
		threshold,
		0.0,
		0.0
	);	
	return marcher;
}

// SURFACE
struct Surface {
    vec3 position;
    vec3 normal;
	vec3 rgb;
};
Surface getSurface(Marcher marcher) {
	// Use the "dist" value from marcher to obtain the surface postion, which can be passed down the pipeline for lighting.
	vec3 position = marcher.origin + marcher.direction * marcher.distance;
	// We can use the surface position to calculate the surface normal using a bit of vector math. I remember having to give long, drawn-out,
	// sleep-inducing talks at uni on implicit surface geometry (or something like that) that involved normals on 3D surfaces and such.
	// I barely remember the content, but I definitely remember there was always this hot chick in the room with a gigantic set of silicons who
	// looked entirely out of place amongst all the nerds... and this was back in the days when those things weren't as common... Um, I forgot
	// where I was going with this.
	// Anyway, check out the function itself. It's a standard, but pretty clever way to get a surface normal on difficult-to-differentiate surfaces.
	vec3 normal = getNormal(position);
	Surface surface = Surface(position, normal, vec3(0.0));
	return surface;
}

// LIGHT
struct Light {
	vec3 color;
	vec3 position;
	vec3 direction;
	vec3 reflected;
	float distance;
	float attenuation;
	float diffuse;
	float specular;
};
Light getLight(vec3 color, vec3 position, Material material, Surface surface, Camera camera) {
	// Light needs to have a position, a direction and a color. Obviously, it should be positioned away from the
	// object's surface. The direction vector is the normalized vector running from the light position to the object's surface point that we're
	// going to illuminate. You can choose any light color you want, but it's probably best to choose a color that works best with the colors
	// in the scene. I've gone for a warmish white.
	// color. Light color. I have it in my head that light globes give off this color, but I swear I must have pulled that information right out of my a... Choose any color you want.
	// position. I've arranged for it to move in a bit of a circle about the xy-plane a couple of units away from the spherical object.
	// direction. Light direction. The point light direction goes from the light's position to the surface point we've hit on the sphere. I haven't normalized it yet, because I'd like to take the length first, but it will be.
	// reflected. The unit-length, reflected vector. Angle of incidence equals angle of reflection, if you remember rudimentary highschool physics, or math.
	// Anyway, the incident (incoming... for want of a better description) vector is the vector representing our line of sight from the light position
	// to the point on the suface of the object we've just hit. We get the reflected vector on the surface of the object by doing a quick calculation
	// between the incident vector and the surface normal. The reflect function is ( ref=incidentNorm-2.0*dot(incidentNorm, surfNormal)*surfNormal ),
	// or something to that effect. Either way, there's a function for it, which is used below.
	// The reflected vector is useful, because we can use it to calculate the specular reflection component. For all intents and purposes, specular light
	// is the light gathered in the mirror direction. I like it, because it looks pretty, and I like pretty things. One of the most common mistakes made
	// with regard to specular light calculations is getting the vector directions wrong, and I've made the mistake more than a few times. So, if you
	// notice I've got the signs wrong, or anything, feel free to let me know.
	// distance. Distance from the light to the surface point.
	// attenuation. Light falloff (attenuation), which depends on how far the surface point is from the light. Most of the time, I guess the falloff rate should be
	// mixtures of inverse distance powers, but in real life, it's far more complicated than that. Either way, most of the time you should simply
	// choose whatever makes the lighting look a little prettier. For instance, if things look too dark, I might decide to make the falloff drop off
	// linearly, without any other terms. In this case, the light is falling off with the square of the distance, and no other terms.
	// diffuse. The object's diffuse value, which depends on the angle that the light hits the object.
	// specular. The object's specular value, which depends on the angle that the reflected light hits the object, and the viewing angle... kind of.
	// specularity. The power of the specularity. Higher numbers can give the object a harder, shinier look.
	const float specularity = 16.0;
	Light light = Light(color, position, vec3(0.0), vec3(0.0), 0.0, 0.0, 0.0, 0.0);
	light.direction = light.position - surface.position;
	light.distance = length(light.direction);
	light.direction /= light.distance; // Normalizing the light-to-surface, aka light-direction, vector.
	light.attenuation = min(1.0 / (0.25 * light.distance * light.distance), 1.0); // Keeps things between 0 and 1.
	light.reflected = reflect(-light.direction, surface.normal);
	light.diffuse = max(0.0, dot(surface.normal, light.direction));
	light.specular = max(0.0, dot(light.reflected, normalize(camera.position - surface.position)));
	light.specular = pow(light.specular, specularity); // Ramping up the specular value to the specular power for a bit of shininess.
	return light;
}
vec3 calcLight (Light light, Material material) {
	// Bringing all the lighting components together to color the screen pixel. By the way, this is a very simplified version of Phong lighting.
	// It's "kind of" correct, and will suffice for this example. After all, a lot of lighting is fake anyway.
	return (material.color * (material.ambient + light.diffuse * material.glossiness) + light.specular * material.shininess) * light.color * light.attenuation;
}

float getRayDistance(Marcher marcher, Camera camera) {
	marcher.distance = 0.0;
	marcher.depth = camera.near; // Ray depth. "start" is usually zero, but for various reasons, you may wish to start the ray further away from the origin.
	for (int i = 0; i < STEPS; i++ ) {
		// Distance from the point along the ray to the nearest surface point in the scene.
		marcher.distance = scene(marcher.origin + marcher.direction * marcher.depth);
        // Irregularities between browsers have forced me to use this logic. I noticed that Firefox was interpreting two "if" statements inside a loop
        // differently to Chrome, and... 20 years on the web, so I guess I should be used to this kind of thing.
        // Anyway, belive it or not, the stop threshold is one of the most important values in your entire application. Smaller numbers are more
        // accurate, but can slow your program down - drastically, at times. Larger numbers can speed things up, but at the cost of aesthetics.
        // Swapping a number, like "0.001," for something larger, like "0.01," can make a huge difference in framerate.
		if ((marcher.distance < marcher.threshold) || (marcher.depth >= camera.far)) {
		    // (rayDepth >= end) - We haven't used up all our iterations, but the ray has reached the end of the known universe... or more than
		    // likely, just the far-clipping-plane. Either way, it's time to return the maximum distance.
		    // (scene.distance < marcher.threshold) - The distance is pretty close to zero, which means the point on the ray has effectively come into contact
		    // with the surface. Therefore, we can return the distance, which can be used to calculate the surface point.
			// I'd rather neatly return the value above. Chrome and IE are OK with it. Firefox doesn't like it, etc... I don't know, or care,
			// who's right and who's wrong, but I would have thought that enabling users to execute a simple "for" loop without worring about what
			// will work, and what will not, would be a priority amongst the various parties involved. Anyway, that's my ramble for the day. :)
			break;
		}
		// We haven't hit anything, so increase the depth by a scaled factor of the minimum scene distance. It'd take too long to explain why
		// we'd want to increase the ray depth by a smaller portion of the minimum distance, but it can help, believe it or not.
		marcher.depth += marcher.distance * marcher.scale;
	}
	// I'd normally arrange for the following to be taken care of prior to exiting the loop, but Firefox won't execute anything before
	// the "break" statement. Why? I couldn't say. I'm not even game enough to put more than one return statement.
	// Normally, you'd just return the rayDepth value only, but for some reason that escapes my sense of logic - and everyone elses, for
	// that matter, adding the final, infinitessimal scene distance value (sceneDist) seems to reduce a lot of popping artifacts. If
	// someone could put me out of my misery and prove why I should either leave it there, or get rid of it, it'd be appreciated.
	if (marcher.distance >= marcher.threshold) {
		marcher.depth = camera.far;
	} else {
		marcher.depth += marcher.distance;
	} 
	// We've used up our maximum iterations. Damn, just a few more, and maybe we could have hit something, or maybe there was nothing to hit.
	// Either way, return the maximum distance, which is usually the far-clipping-plane, and be done with it.
	return marcher.depth;
}

vec3 render() {
	// BACKGROUND
	vec3 background = vec3(0.01, 0.01, 0.01);
	// CAMERA
	Camera camera = getCamera(
		vec3(4.0, 1.0, 4.0), // vec3(4.0 * sin(u_time), 1.0, 4.0 * cos(u_time)), // position
		vec3(0.0, 0.0, 0.0) // target
	);
	// MARCHER
	Marcher marcher = getMarcher(camera);
	marcher.distance = getRayDistance(marcher, camera);
	if (marcher.distance >= camera.far) {
	    // I prefer to do it this way in order to avoid an if-statement below, but I honestly couldn't say whether it's more
	    // efficient. It feels like it would be. Does that count? :)
	    return background;
		//discard; // If you want to return without rendering anything, I think.
	}
	// SURFACE. If we've made it this far, we've hit something. 
	Surface surface = getSurface(marcher);
	if (false) {
		// Just some really lame, fake shading/coloring for the object. You can comment the two lines out with no consequence.
		float bumps = sinusoidBumps(surface.position);
		material.color = clamp(material.color * 1.0 - vec3(1.0, 1.0, 1.0) * bumps, 0.0, 1.0);
	}
	// LIGHT
	Light light = getLight(
		vec3(1.0, 1.0, 1.0), // color
		vec3(1.0), // vec3(1.5 * sin(u_time * 0.5), 0.75 + 0.25 * cos(u_time * 0.5), -1.0), // position
		material,
		surface,
		camera
	);
	// LAMBERT
	surface.rgb = calcLight(light, material);
	// FOG
	surface.rgb = mix(surface.rgb, background, clamp(marcher.distance / 20.0, 0.0, 1.0));
	// Clamping the lit pixel between black and while, then putting it on the screen. We're done. Hooray!
	return clamp(surface.rgb, 0.0, 1.0); // from 0 to 1
}

void main() {
	setVars();

	// vec2 xy = gl_FragCoord.xy / u_resolution.xy;
	vec3 rgb = vec3(0.0);

    // rgb += vec3(vec2(0.0), abs(sin(uv.y)) * .15);
    // rgb += vec3(vec2(0.0), xy.x * 0.1);
	// rgb += vec3(xy.y * 0.1, vec2(0.0));
	// rgb += drawPoint(point, cyan);	

	if (true) {
		rgb += render();
	}

	/*
	if (false) {
		float y = cos(point.x) * .5;
		// float y = step(0.1, st.x);
		// float y = pow(st.x, 10.0);
		// float x = st.x * 20.0;
		// float y = sin(x) + sin(x / 0.5) * 0.5 + sin(x / 0.333) * 0.333 + sin(x / 0.25) * 0.25 + sin(x / 0.2) * 0.2;
    	rgb += drawLine(point, y, green);
	}

	if (false) {
		const int divs = 5;
		int iPos = 0;
		int count = divs * divs * 2; // use when rendering
		vec3 c = vec3(0.0);
		for (int j = 0; j < divs; j++) {
			float t = PI * float(j) / float(divs);
			for (int k = 0; k < divs; k++) {
				iPos = j * divs + k;
				float s = TWO_PI * float(k) / float(divs);
				vec4 p = vec4(
					c.x + .5 * sin(t) * cos(s),
					c.y + .5 * sin(t) * sin(s),
					c.z + .5 * cos(t),
					.0
				);
				p *= matrix;
				rgb += drawPoint(p, blue);
			}
		}
	}

    if (false) {
		rgb += drawPoint(vec4(-0.5,  0.5,  0.0, 1.0), cyan);
		rgb += drawPoint(vec4(-0.5, -0.5,  0.0, 1.0), cyan);
		rgb += drawPoint(vec4(0.0,   0.0,  0.5, 1.0), yellow);
		rgb += drawPoint(vec4(0.0),                   green);
		rgb += drawPoint(vec4(0.0,   0.0, -0.5, 1.0), yellow);
		rgb += drawPoint(vec4(0.5,   0.5,  0.0, 1.0), red);
		rgb += drawPoint(vec4(0.5,  -0.5,  0.0, 1.0), red);
	}
	*/
	gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}