// License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.
// Created by L.Zampetti (C) 2017 

// ROW NUMBERS -87
#extension GL_OES_standard_derivatives : enable

#ifdef GL_ES
	precision highp float;
	precision highp int;
#endif

// CONSTANTS
#define PI              3.14159265359
#define TWO_PI          6.28318530718
#define PI_OVER_TWO     1.57079632679
#define ONE_OVER_PI     0.31830988618
#define EPSILON         0.00100000000
#define BIG_FLOAT       1000000.00000

#define BLUR            8.0
#define BLUR_I          4.0
#define NM_STEPS        5.0

#ifndef u_resolution
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

struct Compass {
	float n, s, w, e;
};

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

mat2 rotate2d(float angle) {
    return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
}

float random(vec3 scale, float seed) {
	/* use the fragment position for a different seed per-pixel */
	return fract(sin(dot(gl_FragCoord.xyz + seed, scale)) * 43758.5453 + seed);
}

vec4 bevel(vec2 uv) {
	//---------------------------------------------
	// 135  90  45
	//    \  |  /
	//180 -  .  - 360/0
	//    /  |  \
	// 225  270  315
	float dir = u_time * 5.; //Degrees
	float dist = 2. + sin(u_time * 2.); //Distance
	float strength = 1.;
	float invert = 1.; //0, 1
	float BnW = 1.; //Black and white? 0, 1
	//Tips: 0 = False, 1 = True.
	//---------------------------------------------
	//Draw out the outputs
	vec4 rgb = vec4(0.);
	rgb *= 0.001; //Make distance smaller
	if (invert < 1.) {
		rgb = vec4(0.5 + ((texture2D(u_texture, uv).rgb - texture2D(u_texture, uv + (vec2(cos(radians(dir)), sin(radians(dir))) * dist)).rgb) * strength), 1.0);
	} else {
		rgb = vec4(0.5 + ((texture2D(u_texture, uv + (vec2(cos(radians(dir)), sin(radians(dir))) * dist)).rgb - texture2D(u_texture, uv).rgb) * strength), 1.0);    
	}
	if (BnW >= 1.) { 
		rgb = vec4((rgb.r + rgb.g + rgb.b) / vec3(3.0), rgb.a);
	}
	return rgb;
}

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
/*
	float dx = dFdx(rgba.a);
	color.r += dx * 10.0;
*/
/*
	float dy = dFdy(rgba.a);
	color.b += dy * 10.0;
*/
	vec2 size = vec2(1.0 / 1024.0);
	vec2 strength = -1.0 + u_mouse.xy / u_resolution.xy * 2.0;
	strength *= vec2(cos(u_time * 10.0), sin(u_time * 10.0));
	vec4 c = texture2D(u_texture, uv);
	vec4 n = texture2D(u_texture, uv + size * vec2(0.0, -10.0) * strength);
	vec4 s = texture2D(u_texture, uv + size * vec2(0.0, 10.0) * strength);
	vec4 e = texture2D(u_texture, uv + size * vec2(10.0, 0.0) * strength);
	vec4 o = texture2D(u_texture, uv + size * vec2(-10.0, 0.0) * strength);
/*
float dx = dFdx(c.a);
rgb.r += dx * 10.0;
*/
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

vec4 circle(vec2 uv) {
	float v = smoothstep(0.985, 0.99, length(uv));
	return vec4(vec3(v), 1.0);
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

vec4 normalMap(vec2 uv) {
	vec3 rgb = vec3(0.5);
	vec2 pixel = vec2(1.0 / 1024.0);	
	for(float i = 1.0; i < NM_STEPS + 1.0; i++) {
		rgb -= (texture2D(u_texture, uv - pixel * i)).a * 1.0 / i;
		rgb += (texture2D(u_texture, uv + pixel * i)).a * 1.0 / i;	
	}
  rgb /= (NM_STEPS / 2.0);
  return vec4(rgb, 1.0);
}

float lumaFromRgb(vec3 rgb) {
	float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
	return luma;
}
vec3 toNormalMap(vec2 uv) {
	vec2 dx = dFdx(uv);
	vec2 dy = dFdy(uv);
	vec3 n = circle(uv + dy * vec2(0.0, -1.0)).rgb;
	vec3 s = circle(uv + dy * vec2(0.0, 1.0)).rgb;
	vec3 w = circle(uv + dx * vec2(-1.0, 0.0)).rgb;
	vec3 e = circle(uv + dx * vec2(1.0, 0.0)).rgb;
	Compass luma = Compass(lumaFromRgb(n), lumaFromRgb(s), lumaFromRgb(w), lumaFromRgb(e));
	float vertical = ((luma.n - luma.s) + 1.0) * 0.5;
	float horizontal = ((luma.w - luma.e) + 1.0) * 0.5;
	return vec3(horizontal, vertical, 1.0);
}

void main() {
	vec2 st = getSt();
	vec2 uv = getUv(st);
	vec4 color = vec4(vec3(0.0), 0.0);		
	// circle
	if (false) {
		color += circle(uv);
	}
	float a = texture2D(u_texture, st).a;
	// neat
	if (false) {
		color = vec4(vec3(1.0 * a), a);
	}
	// bevel
	if (true) {
		color = normalMap(st);
	}
	// blur
	if (false) {
		color = blurred(st, 1.0 / 1024.0); //  * a;
	}
	if(false) {
		color = effect(st);
	}
	gl_FragColor = color;
}
