// Author:
// Title:

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
#define RAD						0.01745329251

// GLOBALS
vec2 st;
vec2 uv;
vec2 mx;
vec4 point;
vec4 normal;
mat4 matrix;

// COLORS
vec3 red = vec3(1.0, 0.0, 0.0);
vec3 green = vec3(0.0, 1.0, 0.0);
vec3 blue = vec3(0.0, 0.0, 1.0);
vec3 yellow = vec3(1.0, 1.0, 0.0);
vec3 magenta = vec3(1.0, 0.0, 1.0);
vec3 cyan = vec3(0.0, 1.0, 1.0);

// VARS
vec2 getSt() {    
	vec2 st = gl_FragCoord.xy / u_resolution.xy;
	st.y *= u_resolution.y / u_resolution.x;
	st.y += (u_resolution.x-u_resolution.y) / u_resolution.x / 2.0;
	return st;
}
vec2 getUv(vec2 st) {
	vec2 uv = -1.0 + st * 2.0;
	return uv;
}
vec2 getMx() {
	return -1.0 + u_mouse / u_resolution.xy * 2.0;
}
vec2 pixel = vec2(1.0) / u_resolution;

// MATH
const mat4 projection = mat4(
	vec4(3.0 / 4.0, 0.0, 0.0, 0.0),
	vec4(     0.0, 1.0, 0.0, 0.0),
	vec4(     0.0, 0.0, 0.5, 0.5),
	vec4(     0.0, 0.0, 0.0, 1.0)
);
mat4 rotation = mat4(
	vec4(1.0,          0.0,         0.0, 	0.0),
	vec4(0.0,  cos(u_time), sin(u_time),  0.0),
	vec4(0.0, -sin(u_time), cos(u_time),  0.0),
	vec4(0.0,          0.0,         0.0, 	1.0)
);
mat4 scale = mat4(
	vec4(4.0 / 3.0, 0.0, 0.0, 0.0),
	vec4(     0.0, 1.0, 0.0, 0.0),
	vec4(     0.0, 0.0, 1.0, 0.0),
	vec4(     0.0, 0.0, 0.0, 1.0)
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

float drawLine(vec2 p, float y, float size) {
	return smoothstep(y - pixel.x * size, y, p.y) - smoothstep(y, y + pixel.y * size, p.y);
}
float drawLine(vec2 p, float y) {
	return drawLine(p, y, 1.0);
}

vec3 drawPoint(vec4 p, vec3 color, float size) {
	vec2 screen = toScreen(p);
	float l = length(uv - screen);
	if (l < 0.003 * size) {
		return color;
	} else {
		return vec3(0.0);
	}	
}
vec3 drawPoint(vec4 p, vec3 color) {
	return drawPoint(p, color, 1.0);	
}

void main() {
    st = getSt();

    gl_FragColor = vec4(vec3(st.x * 0.1), 1.0);

    /*
	uv = getUv(st);
	mx = getMx();
	point = vec4(uv, 0.0, 1.0);
	normal = vec4(0.0, 0.0, 1.0, 1.0);
	matrix = rotationAxis(RAD * 10.0 * -mx.x * cos(u_time), vec3(0.0, 1.0, 0.0));
	matrix *= rotationAxis(RAD * 10.0 * -mx.y * sin(u_time), vec3(1.0, 0.0, 0.0));
	point *= matrix;
	normal *= matrix;
	point.z -= 1.1;

    vec3 rgb = vec3(0.0);
    
    float y = st.x;
    // float y = step(0.1, st.x);
    // float y = pow(st.x, 10.0);
    // float x = st.x * 20.0;
    // float y = sin(x) + sin(x / 0.5) * 0.5 + sin(x / 0.333) * 0.333 + sin(x / 0.25) * 0.25 + sin(x / 0.2) * 0.2;
    
    float line = drawLine(st, y);
    rgb += line;
    
    if (true) {
		rgb += drawPoint(vec4(-0.5,  0.5, 0.0, 1.0), green);
		rgb += drawPoint(vec4(-0.5, -0.5, 0.0, 1.0), green);
		rgb += drawPoint(vec4(0.0),                  blue);
		rgb += drawPoint(vec4(0.5,   0.5, 0.0, 1.0), green);
		rgb += drawPoint(vec4(0.5,  -0.5, 0.0, 1.0), green);				
	}

    gl_FragColor = vec4(clamp(rgb, 0.0, 1.0), 1.0);
    */
}