let ts = `
`;
let shaderA = {
constants: `
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

#define ONE_OVER_TWO_PI 0.15915494309
#define RAD             0.01745329251 // PI / 180
#define F4              0.30901699437 // F4 = (sqrt(5.0) - 1.0) / 4.0    0.309016994374947451
#define G4              0.13819660112 // G4 = (5.0 - sqrt(5.0)) / 20.0
`,
structs: ` 
struct PointLight {
	float distance;
	float decay;
	float shadowBias;
	float shadowRadius;
	int shadow;
	vec2 shadowMapSize;
	vec3 position;
	vec3 color;
};

struct Compass {
	float n, s, w, e;
};

struct Texture {
	vec3 color;
	vec4 offset;
	float strength;
	float bias;
};

struct Flags {
	bool displacement, diffuse, normal, bump, lights, albedo, noise, clouds;
};
`,
uniforms: `
// attribute position: vec3, the vertex itself
// attribute normal: vec3, the normal at the current vertex
// attribute uv: vec2, the texture coord

// uniform mat4 projectionMatrix; // mat4, self explanatory
// uniform mat4 modelMatrix; // mat4, object-to-world matrix
// uniform mat4 viewMatrix; // mat4, world-to-camera matrix
// uniform mat4 modelViewMatrix; // mat4, same as viewMatrix*modelMatrix, or object-to-camera matrix
// uniform mat3 normalMatrix;
// uniform vec3 cameraPosition;

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

uniform vec3 ambient;
uniform vec3 specular;

uniform float glossiness;
uniform float specularity;

varying vec2 vUv;
varying vec3 vecPos;
varying vec3 vecNormal;
varying vec3 vNormal;
varying vec3 vViewPosition;
varying mat4 modelView;

uniform sampler2D diffuseMap;
uniform Texture diffuseObj;
varying vec2 diffuseUv;

uniform sampler2D normalMap;
uniform Texture normalObj;
varying vec2 normalUv;

uniform sampler2D bumpMap;
uniform Texture bumpObj;
varying vec2 bumpUv;

uniform PointLight pointLights[NUM_POINT_LIGHTS];

uniform Flags flags;
`,
math: `
mat2 rotate2d(float angle){
    return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
}
`,
luma: `
float lumaFromRgb(vec3 rgb) {
	float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
	return luma;
}
vec3 rgbAtOffset(sampler2D source, vec2 xy, vec4 offsetRepeat) {
	vec3 rgb = texture2D(source, (xy + offsetRepeat.xy)).rgb;
	return rgb;
}
float lumaAtOffset(sampler2D source, vec2 xy, vec4 offsetRepeat) {
	vec3 rgb = texture2D(source, (xy + offsetRepeat.xy)).rgb;
	float luma = dot(rgb, vec3(0.2126, 0.7152, 0.0722));
	return luma;
}
`,
noise: `
vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
float mod289(float x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}
vec4 permute(vec4 x) {
    return mod289(((x*34.0)+1.0)*x);
}
float permute(float x) {
    return mod289(((x*34.0)+1.0)*x);
}
vec4 invSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}
float invSqrt(float r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}
vec4 grad4(float j, vec4 ip) {
    const vec4 ones = vec4(1.0, 1.0, 1.0, -1.0);
    vec4 p,s;
    p.xyz = floor( fract (vec3(j) * ip.xyz) * 7.0) * ip.z - 1.0;
    p.w = 1.5 - dot(abs(p.xyz), ones.xyz);
    s = vec4(lessThan(p, vec4(0.0)));
    p.xyz = p.xyz + (s.xyz*2.0 - 1.0) * s.www;
    return p;
}
float snoise(vec4 v) {
	const vec4  C = vec4(
		0.138196601125011,  // (5 - sqrt(5))/20  G4
		0.276393202250021,  // 2 * G4
		0.414589803375032,  // 3 * G4
		-0.447213595499958  // -1 + 4 * G4
	);
	vec4 i  = floor(v + dot(v, vec4(F4)) );
	vec4 x0 = v -   i + dot(i, C.xxxx);
	vec4 i0;
	vec3 isX = step( x0.yzw, x0.xxx );
	vec3 isYZ = step( x0.zww, x0.yyz );
	i0.x = isX.x + isX.y + isX.z;
	i0.yzw = 1.0 - isX;
	i0.y += isYZ.x + isYZ.y;
	i0.zw += 1.0 - isYZ.xy;
	i0.z += isYZ.z;
	i0.w += 1.0 - isYZ.z;
	vec4 i3 = clamp( i0, 0.0, 1.0 );
	vec4 i2 = clamp( i0-1.0, 0.0, 1.0 );
	vec4 i1 = clamp( i0-2.0, 0.0, 1.0 );
	vec4 x1 = x0 - i1 + C.xxxx;
	vec4 x2 = x0 - i2 + C.yyyy;
	vec4 x3 = x0 - i3 + C.zzzz;
	vec4 x4 = x0 + C.wwww;
	i = mod289(i);
	float j0 = permute( permute( permute( permute(i.w) + i.z) + i.y) + i.x);
	vec4 j1 = permute( permute( permute( permute (
			i.w + vec4(i1.w, i2.w, i3.w, 1.0 ))
			+ i.z + vec4(i1.z, i2.z, i3.z, 1.0 ))
			+ i.y + vec4(i1.y, i2.y, i3.y, 1.0 ))
			+ i.x + vec4(i1.x, i2.x, i3.x, 1.0 )
	);
    vec4 ip = vec4(1.0/294.0, 1.0/49.0, 1.0/7.0, 0.0);
	vec4 p0 = grad4(j0,   ip);
	vec4 p1 = grad4(j1.x, ip);
	vec4 p2 = grad4(j1.y, ip);
	vec4 p3 = grad4(j1.z, ip);
	vec4 p4 = grad4(j1.w, ip);
	vec4 norm = invSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2, p2), dot(p3,p3)));
	p0 *= norm.x;
	p1 *= norm.y;
	p2 *= norm.z;
	p3 *= norm.w;
	p4 *= invSqrt(dot(p4,p4));
	vec3 m0 = max(0.6 - vec3(dot(x0,x0), dot(x1,x1), dot(x2,x2)), 0.0);
	vec2 m1 = max(0.6 - vec2(dot(x3,x3), dot(x4,x4)            ), 0.0);
	m0 = m0 * m0;
	m1 = m1 * m1;
	return 49.0 * (dot(m0*m0, vec3( dot( p0, x0 ), dot( p1, x1 ), dot( p2, x2 ))) + dot(m1*m1, vec2(dot(p3, x3), dot(p4, x4))));
}
float surface( vec4 coord ) {
	float n = 0.0;
	n += 0.25 * abs(snoise( coord * 4.0 ));
	n += 0.5 * abs(snoise( coord * 8.0 ));
	n += 0.25 * abs(snoise( coord * 16.0 ));
	n += 0.125 * abs(snoise( coord * 32.0 ));
	return n;
}
vec3 clouds(vec2 vUv) {
	float m = 1.0 / (2.0 * PI);
	vec3 color = vec3(1.0, 1.0, 1.0);
	vec2 scale = vec2(2.0, 1.0);
	vec2 p = vUv * scale;
	vec4 n = vec4(
		cos(p.x * 2.0 * PI) * m, 
		cos(p.y * 2.0 * PI) * m, 
		sin(p.x * 2.0 * PI) * m, 
		sin(p.y * 2.0 * PI) * m
	);
	float s = surface(n + u_time * 0.002);
	s = smoothstep(.25, .65, s);
	return color * s;
}
vec3 cloudToNormal() {
	vec2 dx = dFdx(vUv);
	vec2 dy = dFdy(vUv);
	vec3 n = clouds(vUv + dy * vec2(0.0, -1.0));
	vec3 s = clouds(vUv + dy * vec2(0.0, 1.0));
	vec3 w = clouds(vUv + dx * vec2(-1.0, 0.0));
	vec3 e = clouds(vUv + dx * vec2(1.0, 0.0));
	Compass luma = Compass(lumaFromRgb(n), lumaFromRgb(s), lumaFromRgb(w), lumaFromRgb(e));
	float vertical = ((luma.n - luma.s) + 1.0) * 0.5;
	float horizontal = ((luma.w - luma.e) + 1.0) * 0.5;
	return vec3(horizontal, vertical, 1.0);
}
`,
normals: `
vec2 dHdxy_fwd() {
	vec2 xy = mod(bumpUv, 1.0);
	vec2 dSTdx = dFdx(xy);
	vec2 dSTdy = dFdy(xy);
	float Hll = bumpObj.strength * texture2D(bumpMap, xy).x;
	float dBx = bumpObj.strength * texture2D(bumpMap, xy + dSTdx).x - Hll;
	float dBy = bumpObj.strength * texture2D(bumpMap, xy + dSTdy).x - Hll;
	return vec2(dBx, dBy);
}
// BUMP MAP
vec3 perturbNormalArb(vec3 position, vec3 normal, vec2 dHdxy) {
	vec3 vSigmaX = dFdx(position);
	vec3 vSigmaY = dFdy(position);
	vec3 vN = normal; // normalized
	vec3 R1 = cross(vSigmaY, vN);
	vec3 R2 = cross(vN, vSigmaX);
	float fDet = dot(vSigmaX, R1);
	vec3 vGrad = sign(fDet) * (dHdxy.x * R1 + dHdxy.y * R2);
	return normalize(abs(fDet) * normal - vGrad);
}
// NORMAL MAP
// http://hacksoflife.blogspot.ch/2009/11/per-pixel-tangent-space-normal-mapping.html
vec3 perturbNormal2Arb(vec3 viewPos, vec3 normal) {
	vec2 scale = vec2(normalObj.strength);
	vec2 c = mod(normalUv.st, 1.0);
	vec3 q0 = dFdx(viewPos.xyz);
	vec3 q1 = dFdy(viewPos.xyz);
	vec2 st0 = dFdx(c);
	vec2 st1 = dFdy(c);
	vec3 S = normalize(q0 * st1.t - q1 * st0.t);
	vec3 T = normalize(-q0 * st1.s + q1 * st0.s);
	vec3 N = normalize(normal);
	vec3 mapN = texture2D(normalMap, mod(normalUv, 1.0)).xyz * 2.0 - 1.0;
	mapN.xy = scale * mapN.xy;
	mat3 tsn = mat3(S, T, N);
	return normalize(tsn * mapN);
}
// NOISE MAP
vec3 perturbNormalNoise(vec3 normal) {
		vec3 noise = clouds(vUv);
		/*
		noise.x = dFdx(length(noise));
		noise.y = dFdy(length(noise));
		noise.z = sqrt(1.0 - noise.x * noise.x - noise.y * noise.y);
		*/
		normal += (-1.0 * RAD + (2.0 * RAD * noise));
		return normalize(normal);
}
// HEIGHT MAP
vec4 heightToNormal(sampler2D texture, vec2 xy) { 
	vec3 n = rgbAtOffset(texture, xy, vec4(0.0, -1.0, 1.0, 1.0));
	vec3 s = rgbAtOffset(texture, xy, vec4(0.0, 1.0, 1.0, 1.0));
	vec3 w = rgbAtOffset(texture, xy, vec4(-1.0, 0.0, 1.0, 1.0));
	vec3 e = rgbAtOffset(texture, xy, vec4(1.0, 0.0, 1.0, 1.0));
	Compass luma = Compass(lumaFromRgb(n), lumaFromRgb(s), lumaFromRgb(w), lumaFromRgb(e));
	float vertical = ((luma.n - luma.s) + 1.0) * 0.5;
	float horizontal = ((luma.w - luma.e) + 1.0) * 0.5;
	return vec4(horizontal, vertical, 1.0, 1.0);
}
// COLOR MAP
vec3 perturbNormalWithRgb(vec3 normal, vec3 rgb) {
		normal += (-5.0 * RAD + (10.0 * RAD * rgb));
		return normalize(normal);
}
`,
lights: `
vec3 lights(vec3 normal) {
	float ls = 1.0, ss = 1.0;
	vec3 rgb = vec3(ambient);
	PointLight pl;
	for(int i = 0; i < NUM_POINT_LIGHTS; i++) {
		pl = pointLights[i];
		vec3 vector = pl.position - (-vViewPosition);
		vec3 direction = normalize(vector);
	  float shininess = 0.0;
		float lambert = max(dot(direction, normal), 0.0);		
		if (lambert > 0.0) {
				vec3 halfDir = normalize(direction + normalize(vViewPosition));
				float dp = max(dot(halfDir, normal), 0.0);
				if (dp > 0.0) {
					shininess = min(1.0, pow(dp, 16.0));
					shininess = smoothstep(min(0.98, glossiness * specularity), 0.99, pow(dp, 16.0)) * 0.95;					
				}
		}
		// float constant = 1.0;
		// float linear = 1000.00;
		// float quadratic = 0.9;
        // float attenuation = (constant + linear * distance + quadratic * (distance * distance));
		float distance = length(vector);
		float attenuation = 1.0 / (distance / 500.0 - 1.0);
		vec3 L = (lambert * glossiness * pl.color * attenuation);
		vec3 S = (specular * specularity * shininess * attenuation);
		vec3 color = 	L * (1.0 + S) + (S * specularity * .5);
		rgb += color;
	}
	/*
	rgb.r = clamp(0.01, 1.0, rgb.r);
	rgb.g = clamp(0.01, 1.0, rgb.g);
	rgb.b = clamp(0.01, 1.0, rgb.b);
	*/
	return rgb;
}
`,
fragment: `
void main() {
	vec3 rgb = diffuseObj.color;
	if (flags.diffuse) {
		vec4 trgb = texture2D(diffuseMap, mod(diffuseUv, 1.0));
		trgb = mapTexelToLinear(trgb);
		trgb.a = 1.0;
		rgb = mix(rgb, trgb.rgb, diffuseObj.strength);
	}
	float flip = 1.0;
	vec3 normal = normalize(vNormal) * flip;
	if (flags.normal) {
		normal = perturbNormal2Arb(-vViewPosition, normal);
	} else if (flags.bump) {
		normal = perturbNormalArb(-vViewPosition, normal, dHdxy_fwd());
	}
/*
	if (flags.noise) {
		// normal = perturbNormalNoise(normal);
		vec3 trgb = cloudToNormal();
		rgb += trgb * 5.0;
		normal = perturbNormalWithRgb(normal, trgb * 10.0);
	}
	if (flags.clouds) {
		vec3 trgb = clouds(vUv);
		rgb += trgb;
	}
*/
	vec3 light = vec3(0.0);
	// float luma = 0.0;
	if (flags.lights) {
		light = lights(normal);
		rgb *= light;
		// luma = lumaFromRgb(light);
		// c = mix(c, light, light);
		// c *= light; 
		// c += mix(vec3(0.0), light, light);
	}
	if (flags.albedo) {
		float dp = max(dot(normalize(vViewPosition), normal * 1.0), 0.0);
		float albedo = min(1.0, max(1.0 - dp, 0.0));
		albedo = smoothstep(0.39, 0.79, albedo);
		// albedo *= albedo * luma * luma;
		rgb += albedo * smoothstep(0.5, 0.8, light) * specular;
	}
	gl_FragColor = vec4(rgb, 1.0);
}
`,
vertex: `
void main() { 
	diffuseUv = uv * diffuseObj.offset.zw + diffuseObj.offset.xy; 
	normalUv = uv * normalObj.offset.zw + normalObj.offset.xy;
	bumpUv = uv * bumpObj.offset.zw + bumpObj.offset.xy;
	vUv = uv;
	vec3 objectNormal = vec3(normal);
	#ifdef FLIP_SIDED
		objectNormal = -objectNormal;
	#endif
	vec3 transformedNormal = normalMatrix * objectNormal;
	#ifndef FLAT_SHADED // Normal computed with derivatives when FLAT_SHADED
		vNormal = normalize(transformedNormal);
	#endif
	vec3 transformed = vec3(position);
	if (flags.displacement) {
		float x = 1.0 + cos(u_time + transformed.x * 0.1) * .03;
		float y = 1.0 + cos(u_time + transformed.y * 0.2) * .02;
		float z = 1.0 + cos(u_time + transformed.z * 0.3) * .01;
		transformed.x *= x;
		transformed.y *= y;
		transformed.z *= z;
		vNormal.x += (x * RAD); // tofix
		vNormal.y += (y * RAD); // tofix
		vNormal.z += (z * RAD); // tofix
	}
	#ifdef USE_DISPLACEMENTMAP
		transformed += normal * (texture2D(displacementMap, uv).x * displacementObj.strength + displacementObj.bias);
	#endif
	vec4 mvPosition = modelViewMatrix * vec4(transformed, 1.0);
	vViewPosition = -mvPosition.xyz;
	gl_Position = projectionMatrix * mvPosition;
	// gl_Position = projectionMatrix * modelViewMatrix * vec4(vecPos, 1.0);
	vec4 worldPosition = modelMatrix * vec4(transformed, 1.0);
	#ifdef USE_ENVMAP
		#if defined(USE_BUMPMAP) || defined(USE_NORMALMAP) || defined(PHONG)
			vWorldPosition = worldPosition.xyz;
		#else
			vec3 cameraToVertex = normalize(worldPosition.xyz - cameraPosition);
			vec3 worldNormal = inverseTransformDirection(transformedNormal, viewMatrix);
			#ifdef ENVMAP_MODE_REFLECTION
				vReflect = reflect(cameraToVertex, worldNormal);
			#else
				vReflect = refract(cameraToVertex, worldNormal, refractionRatio);
			#endif
		#endif
	#endif
	vecPos = (modelMatrix * vec4(position, 1.0)).xyz;
	vecNormal = (modelMatrix * vec4(normal, 0.0)).xyz;
}
`,
};
class World {
	clock: THREE.Clock;
	scene: THREE.Scene;
	camera: THREE.PerspectiveCamera;
	renderer: THREE.WebGLRenderer;
	light: THREE.PointLight;
	size: Object;
	mouse: Object;
	options: Object;
	constructor(options: Object) {
		let node, scene, camera, controls, renderer, clock, 
				fov, near, far, width, height;
		this.options = {
			id: 'scene',
			object: function(world: World) { },
			lights: function(world: World) { },
			render: function(delta: Number) { },
			params: function(params: Object) { }
		};
		for (var key:String in options) {
        this.options[key] = options[key];
		}
		this.setSize();
		fov = 60;
		near = 1;
		far = 20000; 
		scene = new THREE.Scene();
		// scene.fog = new THREE.Fog(0x363d3d, -1, 3000 );
		camera = new THREE.PerspectiveCamera(fov, this.size.aspect, near, far);
		camera.position.z = 160;  
		camera.position.y = -160;
		camera.lookAt(new THREE.Vector3(0,0,0));    
		renderer = new THREE.WebGLRenderer({
			alpha: true, 
			antialias: true 
		});
		renderer.setSize(this.size.width, this.size.height);
		renderer.shadowMap.enabled = true;
		renderer.shadowMap.type = THREE.PCFSoftShadowMap;
		node = document.getElementById(this.options.id);
		node.appendChild(renderer.domElement);
		controls = new THREE.OrbitControls(camera, renderer.domElement);
		controls.minDistance = 100;
		controls.maxDistance = 500;
		controls.minPolarAngle = 0; // radians
		controls.maxPolarAngle = Math.PI; // radians
		clock = new THREE.Clock();
		this.node = node;
		this.scene = scene;
		this.camera = camera;
		this.renderer = renderer;
		this.clock = clock;
		this.controle = controls;
		this.mouse = { x: 0, y: 0 };
		this.options.objects(this);
		this.options.lights(this);
		this.addListeners();
		this.play();
	}
	play() {
		let world = this;
		function loop(time){
			var delta = world.clock.getDelta();
			world.render(delta);
			window.requestAnimationFrame(loop);
		}
		loop();
	}
	render(delta: Number){
		if (this.controls) {
			this.controls.update();
		}
		this.options.render(delta);
		this.renderer.render(this.scene, this.camera);
	}
	setSize() {
		this.size = this.size || {
			width: 0, height: 0, aspect: 0,
		};
		this.size.width = window.innerWidth;
		this.size.height = window.innerHeight;
		this.size.aspect = this.size.width / this.size.height;
		if (this.renderer) {
			this.renderer.setSize(this.size.width, this.size.height);
		}
		if (this.camera) {
			this.camera.aspect = this.size.aspect;
			this.camera.updateProjectionMatrix();
		}
	}
	setParams(params: Object) {
			try {
				// console.log(this);
				this.params = params;
				this.options.params(params);
			} catch(e) {
				console.log(e.message);
			}
	}
	addListeners () {
		let world = this;
		function handleMouseMove(event) {
			world.mouse = { 
				x: event.clientX, 
				y: event.clientY 
			};
		}
		function handleMouseDown(event) {
		}
		function handleMouseUp(event) {
		}
		function handleTouchStart(event) {
			if (event.touches.length > 1) {
				event.preventDefault();
				mousePos = {
					x: event.touches[0].pageX, 
					y: event.touches[0].pageY
				};
			}
		}
		function handleTouchEnd(event) {
			mousePos = {
				x: scene.size.width / 2, 
				y: windowHalfY
			};
		}
		function handleTouchMove(event) {
			if (event.touches.length == 1) {
				event.preventDefault();
				mousePos = {
					x: event.touches[0].pageX, 
					y: event.touches[0].pageY
				};
			}
		}
		function onWindowResize() {
			world.setSize();
		}
		window.addEventListener('resize', onWindowResize, false);
		document.addEventListener('mousemove', handleMouseMove, false);
		document.addEventListener('mousedown', handleMouseDown, false);
		document.addEventListener('mouseup', handleMouseUp, false);
		document.addEventListener('touchstart', handleTouchStart, false);
		document.addEventListener('touchend', handleTouchEnd, false);
		document.addEventListener('touchmove',handleTouchMove, false);
	}
}
class Geopos {
	u: Number;
	v: Number;
	w: Number;
	static toGeopos = function(vector: THREE.Vector3) {
		let radius = vector.length();
		let lat = Math.acos(vector.y / radius); //theta
		let lon = Math.atan(vector.x / vector.z); //phi
		return new Geopos(lat, lon, radius);
	}
	static toVector = function(geopos: Geopos) {
		const rad = Math.PI / 180;
    let lat = geopos.u * 2;
		let lon = geopos.v + 180;
		let radius = geopos.w;
		let x = radius * Math.cos(lat * rad) * Math.cos(lon * rad)
		let y = radius * Math.cos(lat * rad) * Math.sin(lon * rad)
		let z = radius * Math.sin(lat * rad)
		/*
		lat = asin(z / radius);
   	lon = atan2(y, x);
		*/
		/*
		let phi   = lat * rad;
		let theta = lon * rad;
		let sinPhi = Math.sin(phi);
		let cosPhi = Math.cos(phi);
		let sinTheta = Math.sin(theta);
		let cosTheta = Math.cos(theta);
		let x = -radius * sinPhi * cosTheta;
		let z = radius * sinPhi * sinTheta;
		let y = radius * cosPhi;
		*/
		return new THREE.Vector3(x, y, z);
	}
	constructor(u:Number = 0, v:Number = 0, w:Number = 0) {
		this.u = u;
		this.v = v;
		this.w = w;
	}
	randomize(radius:Number) {
		this.u = 90 - Math.random() * 180;
		this.v = -180 + Math.random() * 360;
		this.w = radius;
		return this;
	}
	setPosition(du:Number, dv:Number, vector: THREE.Vector3) {
		this.u = 90 + (((this.u + 90) + du) % 180);
		this.v = -180 + (((this.v + 180) + dv) % 360);
		let position = Geopos.toVector(this);
		vector.set(position.x, position.y, position.z);
	}
}
let shader, uniforms, group, sphere, lights = [];
let world = new World({
	id: 'world',
	objects: function(world: World) {
		group = new THREE.Group();
		group.rotation.y = -Math.PI / 180 * 40;
		let diffusePath = 'http://extranet.wslabs.it/ios/terrain.jpg';
		// let diffusePath = 'https://s-media-cache-ak0.pinimg.com/originals/72/59/d9/7259d9158be0b7e8c62c887fac57ed81.png';
		let normalPath = 'https://cdna.artstation.com/p/assets/images/images/002/873/096/large/luc-as-seamless-nrm.jpg';
		// let normalPath = 'https://filterforge.com/filters/12878-normal.jpg';
		// let normalPath = 'https://www.filterforge.com/filters/634-normal.jpg';
		// let normalPath = 'https://s3.amazonaws.com/docs.knaldtech.com/docuwiki/sci_fi_normal.jpg';
		shader = new THREE.ShaderMaterial({
			uniforms: THREE.UniformsUtils.merge([
				THREE.UniformsLib['lights'], {
					//
					u_time: { value: 1.0 },
					u_resolution: { value: new THREE.Vector2() },
					//
					ambient: { value: new THREE.Color(0x000011) },
					specular: { value: new THREE.Color(0xffffdd) },	
					//
					glossiness: { value: 1.0 },
					specularity: { value: 1.0 },				
					//
					diffuseMap : { value: null },
					diffuseObj: {
						value: {
							color: new THREE.Color(0xdddddd),
							strength: 0.7,
							offset: new THREE.Vector4(0,0,4,2),
						}
					},
					//
					normalMap : { value: null },
					normalObj: {
						value: {
							color: new THREE.Color(0xdddddd),
							strength: 0.2,
							offset: new THREE.Vector4(0,0,8,4),
						}
					},
					//
					bumpMap : { value: null },
					bumpObj: {
						value: {
							color: new THREE.Color(0xdddddd),
							strength: 0.8,
							offset: new THREE.Vector4(0,0,8,4),
						}
					},
					//
					flags: {
						value: { 
							displacement: true,
							diffuse: false,
							normal: false,
							bump: false,
							lights: true,
							albedo: false,
							noise: false,
							clouds: false
						},
					}
				}
			]),
			vertexShader: 	shaderA.constants + 
											shaderA.structs + 
											shaderA.uniforms + 
											shaderA.vertex,
			fragmentShader: shaderA.constants + 
											shaderA.structs + 
											shaderA.uniforms + 
											shaderA.luma + 
											shaderA.noise + 
											shaderA.normals +
											shaderA.lights +
											shaderA.fragment,
			lights: true,
			transparent: true,
		});
		let loader = new THREE.TextureLoader();
		loader.crossOrigin = '';
		loader.load('https://crossorigin.me/' + diffusePath, function (texture) {
			shader.uniforms.diffuseMap.value = texture;
		});	
		loader.load('https://crossorigin.me/' + normalPath, function (texture) {
			shader.uniforms.normalMap.value = texture;
			shader.uniforms.bumpMap.value = texture;
		});
		let geometry = new THREE.SphereGeometry(80, 60, 60);
		sphere = new THREE.Mesh(geometry, shader);
		sphere.rotation.x = - Math.PI / 2;
		sphere.receiveShadow = true;
		sphere.castShadow = true;
		group.add(sphere);
		world.scene.add(group);
	},
	lights: function(world: World) {
		let colors = [0xff0000, 0x00ff00, 0x0000ff]; 
		// let colors = [0xffffcc];
		for (let color of colors) {
				let light = new THREE.PointLight(color, 1, 0, 1);
				light.geopos = new Geopos().randomize(1000);
				world.scene.add(light);
				lights.push(light);
				let helper = new THREE.PointLightHelper(light, 10.0);
				world.scene.add(helper);
		}
	},
	render: function(delta: Number = 0) {
		for (let light of lights) {
				light.geopos.setPosition(0.3, 0.3, light.position);
		}
		sphere.rotation.y -= 0.002;
		shader.uniforms.u_time.value += delta * 12.0;
		// shader.needsUpdate = true;
	},
	params: function(params) {
		for (let flag in shader.uniforms.flags.value) {	
			shader.uniforms.flags.value[flag] = params[flag];
		}
		shader.uniforms.glossiness.value = params.glossiness;
		shader.uniforms.specularity.value = params.specularity;
		// shader.needsUpdate = true;
	},
});
let params = {
    displacement: false,
    diffuse: true,
    normal: true,
    bump: false,
    noise: false,
    clouds: false,
    albedo: true,
    lights: true,
    specularity: 0.8,
    glossiness: 0.8,
    randomize: function () {
        for(let i = 0; i < gui.__controllers.length; i++) {
            let c = gui.__controllers[i];
            if (c.__min) {
                let value = c.__min + (c.__max - c.__min) * Math.random();
                this[c.property] = value;
                c.updateDisplay();
            }
        }
        world.setParams(this);
    },
};
let gui = function datgui() {
    let gui = new dat.GUI();
    gui.closed = true;
    let keys = [];
    for (let param in params) {
        keys.push(param);
    }
    for (let param of keys) {
        // console.log(param);
        let p;
        if (typeof params[param] == 'number') {
            p = gui.add(params, param, 0.0, 1.0);
        } else {
            p = gui.add(params, param);
        }
        p.onChange(function(newValue) {
            world.setParams(params);
        });
    }
    /*
    var f2 = gui.addFolder('Light');
    f2.add(params, 'growthSpeed');
    f2.add(params, 'maxSize');
    f2.add(params, 'message');
    f2.open();
    */
    return gui;
}();
world.setParams(params);