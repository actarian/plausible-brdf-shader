// Author:
// Title:
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

// GLOBALS
vec2 st;
vec2 uv;
vec2 vp;
vec2 mx;
vec2 pixel;
vec4 point;
vec4 normal;
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
		matrix = rotationAxis(RAD * 360.0 * u_time * .2, vec3(1.0, 1.0, 1.0));
		point *= matrix;
	}
	/*
	normal = vec4(0.0, 0.0, 1.0, 1.0);
	normal *= matrix;
	*/
	// normal = getNormal();
	// point.z = 1.;
	// vp = toScreen(point);
	// pixel = vec2(1.0) / u_resolution * 100.0;
	pixel = vec2(0.002);
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

void main() {
	setVars();

	vec2 xy = gl_FragCoord.xy / u_resolution.xy;

	vec3 rgb = vec3(0.0);
    // rgb += vec3(vec2(0.0), abs(sin(uv.y)) * .15);
    rgb += vec3(vec2(0.0), xy.x * 0.1);
	rgb += vec3(xy.y * 0.1, vec2(0.0));

	// rgb += drawPoint(point, cyan);	

	if (false) {
		float y = point.x * 1.; // cos(point.x) * .5;
		// float y = step(0.1, st.x);
		// float y = pow(st.x, 10.0);
		// float x = st.x * 20.0;
		// float y = sin(x) + sin(x / 0.5) * 0.5 + sin(x / 0.333) * 0.333 + sin(x / 0.25) * 0.25 + sin(x / 0.2) * 0.2;
    	rgb += drawLine(point, y, green);
	}

/*
x=origin.x+radius*cos(rotation.y)*cos(rotation.x)
y=origin.y+radius*sin(rotation.x)
z=origin.z+radius*sin(rotation.y)*cos(rotation.x)
*/
	vec3 c = vec3(0.0);
	for (float s = 0.0; s < PI; s += 10.0 * RAD) {
		for (float t = 0.0; t < TWO_PI; t += 10.0 * RAD) {
			vec4 p = vec4(
				c.x + .5 * cos(s) * cos(t + u_time),
				c.y + .5 * sin(s),
				c.z + .5 * sin(s) * cos(t + u_time),
				.0
			);
			// rgb += step(length(toScreen(p).xy - uv), 0.01); 
			rgb += drawPoint(p, blue);
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

	gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
}