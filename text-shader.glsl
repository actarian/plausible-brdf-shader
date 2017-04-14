// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Created by L.Zampetti (C) 2017 

// ROW NUMBERS -87
#extension GL_OES_standard_derivatives : enable
#ifdef GL_ES	
	precision mediump float;
#endif

#define PI						3.14159265359
#define RAD						0.01745329251
#define STEPS					2.0 // 1 - 10
#define AMBIENT				0.4 // 0 - 1
#define GLOSSINESS		0.9 // 0 - 1
#define SPECULARITY		0.9 // 0 - 1
#define ROUNDNESS			0.6 // 0 - 1

#ifdef iResolution
    #define u_resolution    iResolution
    #define u_mouse         iMouse
    #define u_time          iGlobalTime
    #define u_texture       iChannel0
#else
    uniform vec2 u_resolution;
    uniform vec2 u_mouse;
    uniform float u_time;
		uniform float u_size;
    uniform sampler2D u_texture;
#endif

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

// NOISE
float rand2 (in vec2 p) { 
    return fract(sin(dot(p.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

float noise2(vec2 e) {
    vec2 p = e + (u_time / 2.0);
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f*f*(3.0-2.0*f);
    return mix( mix( rand2( i + vec2(0.0,0.0) ), 
                     rand2( i + vec2(1.0,0.0) ), u.x),
                mix( rand2( i + vec2(0.0,1.0) ), 
                     rand2( i + vec2(1.0,1.0) ), u.x), u.y);
}

float random(vec3 scale, float seed) {
	return fract(sin(dot(gl_FragCoord.xyz + seed, scale)) * 43758.5453 + seed);
}

const float BLUR = 8.0, BLUR_I = 4.0;
vec4 blurred(vec2 uv, float delta) {
	float a = 0.0;
	const float i = BLUR_I;
	for (float x = -BLUR; x <= BLUR; x += i) {
		for (float y = -BLUR; y <= BLUR; y += i) {
			vec4 rgb = texture2D(u_texture, uv + vec2(delta * x, delta * y));
			a += rgb.a; // - ((BLUR + x) / BLUR) - ((BLUR + y) / BLUR);
		}
	}
	a /= (BLUR * BLUR);
	a *= i;
	// a = smoothstep(abs(cos(u_time * 3.0)), 1.0, a);
	return vec4(vec3(a), 1.0);
}

vec4 effect(vec2 uv) {
	vec4 rgb = vec4(0.0);	
	vec2 size = vec2(1.0 / u_size);
	vec2 strength = -1.0 + u_mouse/u_resolution * 2.0;
	strength *= vec2(cos(u_time * 10.0), sin(u_time * 10.0));
	vec4 c = texture2D(u_texture, uv);
	vec4 n = texture2D(u_texture, uv + size * vec2(0.0, -10.0) * strength);
	vec4 s = texture2D(u_texture, uv + size * vec2(0.0, 10.0) * strength);
	vec4 e = texture2D(u_texture, uv + size * vec2(10.0, 0.0) * strength);
	vec4 o = texture2D(u_texture, uv + size * vec2(-10.0, 0.0) * strength);
/*
	vec2 modd = vec2(mod(uv.x, 1.0), mod(uv.y, 1.0));
	vec4 bev = texture2D(u_texture, modd);
	float cc = mix(1.0 - 2.0 * (1.0 - bev.a) * (1.0 - bev.a), 2.0 * bev.a * bev.a, step(bev.a, 0.5));
	// fixed4 c = tex2D(_MainTex, IN.texcoord) * IN.color;
  // float bev = texture2D(u_texture, half2((uv.x * uv.z) % 1, (uv.y * uv.w) % 1));
	// c = lerp(1 - 2 * (1 - c) * (1 - bev), 2 * c * bev, step(c, 0.5));
  // c.rgb *= c.a;
	// return c;
*/
	float alpha = 1.0;
	// alpha = fract(clamp(0.1 * uv.y, 0.5 * uv.x, c.a)) * 2.0;	
	rgb.a = c.a;
	rgb.r += n.a * alpha;
	rgb.b += e.a * alpha;
	rgb.g += o.a * alpha;
	rgb.r += s.a * alpha;
	rgb.g += s.a * alpha;
	rgb.rgb *= fract((cos(u_time / 20.0 + uv.x) + sin(u_time / 20.0 + uv.y)) * 20.0);
	return rgb;
}

struct Compass {
	float n, s, w, e;
};

vec3 bevel() {
	vec3 rgb = vec3(0.5);
	vec2 pixel = vec2(1.0 / 1024.0);	
	for(float i = 1.0; i < STEPS + 1.0; i++) {
		rgb -= (texture2D(u_texture, st - pixel * i)).a * 1.0 / i;
		rgb += (texture2D(u_texture, st + pixel * i)).a * 1.0 / i;	
	}
  rgb /= (STEPS / 2.0);
  return rgb;
}

vec3 bevelNormal() {
	vec2 pixel = vec2(1.0 / 512.0);	
	Compass luma = Compass(0.5, 0.5, 0.5, 0.5);
	for(float i = 1.0; i < STEPS + 1.0; i++) {
		luma.n += (texture2D(u_texture, st + pixel * vec2(0.0, -1.0) * i)).a / i;
		luma.s += (texture2D(u_texture, st + pixel * vec2(0.0, 1.0) * i)).a / i;
		luma.w += (texture2D(u_texture, st + pixel * vec2(-1.0, 0.0) * i)).a / i;
		luma.e += (texture2D(u_texture, st + pixel * vec2(1.0, 0.0) * i)).a / i;
	}
	float horizontal = ((luma.w - luma.e) + 1.0) * 0.5;
	float vertical = ((luma.n - luma.s) + 1.0) * 0.5;
	return vec3(horizontal, vertical, 1.0);
}

vec2 toScreen(vec4 p) {
	vec4 screen = projection * scale * p;
	float perspective = screen.z * 0.5 + 1.0;
  screen /= perspective;
	return screen.xy;
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

float drawLine(vec2 p, float fn, float size) {
	vec2 pixel = vec2(1.0, 1.0) / u_resolution;
    return smoothstep(fn - pixel.x * size, fn, p.y) - smoothstep(fn, fn + pixel.y * size, p.y);
}

float drawLine(vec2 p, float fn) {
	return drawLine(p, fn, 100.0);
}

vec3 lambert(vec4 light, vec3 color) {
	vec4 n = normal;
	// deform normal
	if (true) {
		float a = RAD * 180.0 * ROUNDNESS;
		n *= rotationAxis(a, color);
	}
	if (false) {
		float a = (cos(uv.x) - sin(uv.y)) * RAD * 60.0;
		n *= rotationAxis(a, vec3(1.0, 1.0, 0.0));
	}
	vec3 normal = n.xyz;
	normal = normalize(normal);
	vec3 rgb = vec3(0.0);
	vec3 ambient = vec3(AMBIENT);
	vec3 diffuse = vec3(0.5, 0.5, 0.6);
	vec3 specular = vec3(0.8, 0.7, 0.7);
	vec3 view = point.xyz;
	vec3 vector = light.xyz - view;
	vec3 direction = normalize(vector);
	float shininess = 0.0;
	float lambert = max(dot(direction, normal), 0.0);
	if (lambert > 0.0) {
		vec3 halfDir = normalize(direction + normalize(-view));
		float halfProduct = max(dot(halfDir, normal), 0.0);
		if (halfProduct > 0.0) {
			shininess = clamp(0.0, 1.0, pow(halfProduct, 16.0));
			shininess = smoothstep(min(0.98, GLOSSINESS * SPECULARITY), 0.99, shininess) * 0.95;					
		}
	}
	float distance = length(vector);
	float attenuation = 1.0 - pow(distance, 2.0) * .1;
	vec3 L = (lambert * GLOSSINESS * diffuse * attenuation);
	vec3 S = (specular * SPECULARITY * shininess * attenuation);
	rgb += ambient + (L * (1.0 + S) + (S * SPECULARITY * .5));
	return rgb;
}

vec2 getSt() {
	vec2 st = gl_FragCoord.xy/u_resolution.xy;
	st.y *= u_resolution.y/u_resolution.x;
	st.y += (u_resolution.x-u_resolution.y) / u_resolution.x / 2.0;
	return st;
}

vec2 getUv(vec2 st) {
	vec2 uv = -1.0 + st * 2.0;
	return uv;
}

vec2 getMx() {
	return -1.0 + u_mouse/u_resolution.xy * 2.0;
}

void main() {
	st = getSt();
	// dx = dFdx(st);
	// dy = dFdy(st);
	uv = getUv(st);
	mx = getMx();
	point = vec4(uv, 0.0, 1.0);
	normal = vec4(0.0, 0.0, 1.0, 1.0);
	matrix = rotationAxis(RAD * 10.0 * -mx.x * cos(u_time), vec3(0.0, 1.0, 0.0));
	matrix *= rotationAxis(RAD * 10.0 * -mx.y * sin(u_time), vec3(1.0, 0.0, 0.0));
	point *= matrix;
	normal *= matrix;
	point.z -= 1.1;

	// deform texture
	st.xy = (1.0 + toScreen(point * 0.8)) / 2.0;

	vec4 color = vec4(vec3(0.0), 0.0);	
	vec3 light = vec3(0.0, 0.0, 0.0);
	
	// bevel
	if (false) {
		color = vec4(bevel(), 1.0);
	}
	// bevelNormal
	if (true) {
		color = vec4(bevelNormal(), 1.0);
	}
	// gl_FragColor = color;
	vec4 lp = vec4(0.0);
	// light 
	if (true) {
		lp = vec4(vec3(0.5), 1.0) * rotationAxis(u_time * 3.0, vec3(-1.0, 1.0, 1.0));
		// lambert
		if (true) {
			vec4 t = texture2D(u_texture, st);
			color.rgb = t.rgb * lambert(lp, color.rgb) * t.a;
		}
	}
	// goldie
	if (false) {
		float a = texture2D(u_texture, st).a;
		vec4 y = vec4(yellow * a * 0.7, a);
		color = (color + y) / 3.0 + (color * y);
	}
	// blur
	if (false) {
		color += blurred(st, 1.0 / u_size); //  * a;
	}
	if(false) {
		color += effect(st);
	}
	// checkpoint
	if (false) {
		color.rgb += drawPoint(vec4(-0.5,  0.5, 0.0, 1.0), green);
		color.rgb += drawPoint(vec4(-0.5, -0.5, 0.0, 1.0), green);
		color.rgb += drawPoint(vec4(0.0), blue);
		color.rgb += drawPoint(vec4(0.5,   0.5, 0.0, 1.0), green);
		color.rgb += drawPoint(vec4(0.5,  -0.5, 0.0, 1.0), green);				
	}
	// drawlight
	if (false) {
		color.rgb += drawPoint(lp, red, 2.0);
	}
	gl_FragColor = color;
}