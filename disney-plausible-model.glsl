// ROW NUMBERS -87
// References: http://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.0pdf

// CONSTANTS
#define PI          3.14159
#define TWO_PI      6.28318
#define PI_OVER_TWO 1.570796
#define ONE_OVER_PI 0.318310
#define GR          1.61803398
#define EPSILON     0.001
#define BIG_FLOAT   1000000.0

// MATERIAL DEFINES
#define SPHERE_MATL 1.0
#define PLANE_MATL  2.0
#define COLS        3
#define ROWS        3

// SWITCHES
#define FRESNEL_HORNER_APPROXIMATION 1

// STRUCTS
struct Surface {
    vec3 point;
    vec3 normal;
    vec3 color;
    float roughness;
    float specular;
    float metallic;
    float selfShadow;
};

struct Brdf {
    // viewDir is the view direction vector
    vec3 viewDir;
    // The half vector of a microfacet model 
    vec3 halfDir;
    // cos(theta_h) - theta_h is angle between half vector and normal
    float costh; 
    // cos(theta_d) - theta_d is angle between half vector and light dir / view dir
    float costd;      
    // cos(theta_l) - theta_l is angle between the light vector and normal
    float costl;
    // cos(theta_v) - theta_v is angle between the viewing vector and normal
    float costv;
};

struct CameraData {
    vec3 position;
    vec3 direction;
    vec2 st;
};

struct Camera {
    vec3 position;
    vec3 target;
};

struct Light {
    vec3 direction;
    vec3 color;
};

struct Global {
    Camera camera;
    Light light;
    float time;
    vec4 debug;    
};

struct Flags {
    bool environment;
    bool occlusions;
    bool shadows;
    bool post;
};

// GLOBALS
Flags flags = Flags(false, false, false, true); // environment, 
Global global = Global(Camera(vec3(0.0), vec3(0.0)), Light(vec3(0.5, 0.5, 0.0), vec3(1.0)), 0.0, vec4(0.0));

// UTILITIES
vec3 mergeobjs(vec3 a, vec3 b) { 
    return mix(b, a, step(a.x, b.x)); 
}

float uniondf(float a, float b) { 
    return min(a, b); 
}
float intersdf(float a, float b) { 
    return max(a, b); 
}
float diffdf(float a, float b) { 
    return max(a, -b); 
}

bool inbounds(vec3 p, vec3 bounds) {
    return all(lessThanEqual(abs(p), 0.5 * bounds));
}

// XXX: To get around a case where a number very close to zero can result in 
// eradic behavior with sign, we assume a positive sign when a value is 
// close to 0.0
float zeroTolerantSign(float value) {
    float s = 1.0;
    if (abs(value) > EPSILON) {
        s = sign(value);
    }
    return s;   
}
// Returns a vec2 where:
//   result.x = 1.0 or 0.0 if there is a hit
//   result.y = t such that position + t * dir = hit point
vec2 intersectDSPlane(vec3 o, vec3 dir, vec3 pn, vec3 po) {
    float dirDotN = dot(dir, pn);
    // if the ray direction is parallel to the plane, let's just treat the 
    // ray as intersecting  * really *  far off, which will get culled as a
    // possible intersection.
    float denom = zeroTolerantSign(dirDotN) * max(abs(dirDotN), EPSILON);
    float t = min(BIG_FLOAT, -dot(pn, (o - po)) / denom);    
    return vec2(step(EPSILON, t), t);
}
// Returns the ray intersection distance (assumes rd is normalized) to the 
// box.  If the ray originates inside the box, then a t of zero is returned.
// if no intersection takes place, BIG_FLOAT is returned.
float intersectBox(vec3 ro, vec3 rd, vec3 bounds) {
    // XXX: In need of optimization
    float d = BIG_FLOAT;
    if (inbounds(ro, bounds)) {
        d = 0.0;
    } else {
        vec3 srd = sign(rd);
        // Only try to intersect the planes that have normals that are 
        // opposing the ray direction.  Saves us from testing the other 3 
        // walls.  We get away with this since we already handled the case 
        // where the ray originates in the box.
        vec2 rx = intersectDSPlane(ro, rd, vec3( -srd.x, 0.0, 0.0), vec3(0.5 * bounds.x * -srd.x, 0.0, 0.0));
        if (rx.x > 0.5 && inbounds(ro + rd * (rx.y + EPSILON), bounds)) {
            d = min(d, rx.y);
        }
        vec2 ry = intersectDSPlane(ro, rd, vec3(0.0, -srd.y, 0.0), vec3(0.0, 0.5 * bounds.y * -srd.y, 0.0));
        if (ry.x > 0.5 && inbounds(ro + rd * (ry.y + EPSILON), bounds)) {
            d = min(d, ry.y);
        }
        vec2 rz = intersectDSPlane(ro, rd, vec3(0.0, 0.0, -srd.z), vec3(0.0, 0.0, 0.5 * bounds.z * -srd.z));
        if (rz.x > 0.5 && inbounds(ro + rd * (rz.y + EPSILON), bounds)) {
            d = min(d, rz.y);
        }
    }
    return d;
}

// DISTANCE FIELDS
float spheredf(vec3 pos, float r) {
    return length(pos) - r;
}

float planedf(vec3 pos, float yoffset) {
    return abs(pos.y + yoffset);
}
// SCENE MARCHING
vec3 ballsobj(vec3 p, float r) {    
    // XXX: could use some optimization
    vec3 obj = vec3(BIG_FLOAT, SPHERE_MATL, 0.0);
    for (int j = 0; j < ROWS; j++) {
        float xoff = float(ROWS) * 2.1 * r * (float(j) / float(ROWS - 1) - 0.5);
        for (int i = 0; i < COLS; i++) {
            float zoff = float(COLS) * 2.1 * r * (float(i) / float(COLS - 1) - 0.5);
            float t = spheredf(p - vec3(xoff, 0.0, zoff), r);
            obj.z = mix(obj.z, float(i + j * COLS), step(t, obj.x));
            obj.x = min(obj.x, t);
        }
    }    
    return obj;
}
vec3 planeobj(vec3 p, float yoffset) {
    return vec3(planedf(p, yoffset), PLANE_MATL, 1.0);
}
vec3 scenedf(vec3 pos, vec3 rd) {
    vec3 obj = vec3(100.0, -1.0, -1.0);
    float r = 0.5;
    float tbox = 0.0;
    if (dot(abs(rd), vec3(1.0)) > EPSILON) {
        float rbuffer = 2.1 * r;
        float occlusionBuffer = 1.0;
        vec3 bounds = rbuffer * vec3(float(ROWS + 1), 1.0, float(COLS + 1));
        // Add a buffer to the bounding box to account for the 
        // ambient occlusion marching.
        bounds += occlusionBuffer;    
        tbox = intersectBox(pos, rd, bounds);
    }    
    if (tbox < 10.0) {
        vec3 bobj = ballsobj(pos + rd * tbox, r);
        // Add the distance to the bounding box
        bobj.x += tbox;
        obj = mergeobjs(obj, bobj);
    }    
    // vec3 bobj = ballsobj(pos, r);
    // obj = mergeobjs(obj, bobj);
    obj = mergeobjs(obj, planeobj(pos, 0.5));
    return obj;
}

// DIST MARCH
#define DM_STEPS 60
#define DM_MAX 50.0
vec3 getDistanceMarch(vec3 ro, vec3 rd, float maxd) {
    float dist = 0.01;
    vec3 res = vec3(0.0, -1.0, -1.0);
    for (int i=0; i < DM_STEPS; i++) {
        if (abs(dist) < EPSILON || res.x > maxd) {
            continue;
        }
        // advance the distance of the last lookup
        res.x += dist;
        vec3 dfresult = scenedf(ro + res.x * rd, rd);
        dist = dfresult.x;
        res.yz = dfresult.yz;
    }
    if (res.x > maxd) {
        res.y = -1.0; 
    }
    return res;
}

// SOFT SHADOWS
#define SS_STEPS 40
#define SS_SIZE 0.1
float getSoftShadowStrength(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float shadow = 1.0;
    float t = mint;
    for (int i = 0; i < SS_STEPS; i++) {
        if (t < maxt) {
            float h = scenedf(ro + rd * t, rd).x;
            shadow = min(shadow, k * h / t);
            t += SS_SIZE;
        }
    }
    return clamp(shadow, 0.00, 1.0);
}

// AMBIENT OCCLUSION
#define AO_STEPS 5
#define AO_SIZE 0.14
#define AO_SCALE 0.35
float getOcclusionStrength(vec3 p, vec3 n) {
    float occlusion = 0.0;
    float scale = 1.0;
    for (int i = 0; i < AO_STEPS; i++) {
        float step = 0.01 + AO_SIZE * float(i);
        vec3 p = n * step + p;        
        float d = scenedf(p, n).x;
        occlusion += -(d - step) * scale;
        scale *= AO_SCALE;
    }    
    return clamp(occlusion, 0.0, 1.0);
}

// SHADING
void setMaterial(float matid, float surfaceid, inout Surface surface) {
    if (matid - 0.5 < SPHERE_MATL) { 
        surface.color = vec3(1.0, 0.35, 0.5); 
        float ballrow = floor(surfaceid / float(COLS));
        surface.roughness = mod(surfaceid, float(COLS)) / float(COLS - 1);
        surface.metallic = floor(surfaceid / float(COLS)) / float(ROWS - 1);
        surface.specular = 0.8;
    } else if (matid - 0.5 < PLANE_MATL) {
        vec4 pavem = texture2D(iChannel2, 0.1 * surface.point.xz);
        surface.color = vec3(.1 * smoothstep(0.6, 0.3, pavem.r));
        surface.metallic = 0.0;
        surface.roughness = 0.6;
        surface.specular = 0.02; // .05        
        surface.normal.xz += 0.1 * pavem.bg;
        surface.normal = normalize(surface.normal);
        surface.selfShadow = 0.3 * (1.0 - smoothstep(0.4, 0.8, pavem.g));
    }
}

float getDistributionStrength(Brdf brdf, float roughness) {
    // D(h) factor
    // using the GGX approximation where the gamma factor is 2.0
    // Clamping roughness so that a directional light has a specular
    // response. A roughness of perfectly 0 will create light singularities.
    float alpha = roughness * roughness;
    float denom = brdf.costh * brdf.costh * (alpha * alpha - 1.0) + 1.0;
    float distribution = (alpha * alpha) / (PI * denom * denom); 
    // using the GTR approximation where the gamma factor is generalized
    // float gamma = 1.0;
    // float sinth = length(cross(surface.normal, brdf.halfDir));
    // float distribution = 1.0 / pow(alpha * alpha * brdf.costh * brdf.costh + sinth * sinth, gamma);
    return distribution;
}

float getGeometryStrength(Brdf brdf, float roughness) {    
    // G(h,l,v) factor    
    float k = roughness / 2.0;
    float Gv = step(0.0, brdf.costv) * (brdf.costv / (brdf.costv * (1.0 - k) + k));
    float Gl = step(0.0, brdf.costl) * (brdf.costl / (brdf.costl * (1.0 - k) + k));
    float geometry = Gl * Gv;
    return geometry;
}

float pow5(float v){
    float tmp = v * v;
    return tmp * tmp * v;
}
vec3 getFresnelColor(Brdf brdf, Surface surface) {
    // F(h,l) factor
    vec3 F0 = surface.specular * mix(vec3(1.0), surface.color, surface.metallic);    
#if FRESNEL_HORNER_APPROXIMATION
    vec3 F = F0 + (1.0 - F0) * exp2((-5.55473 * brdf.costd - 6.98316) * brdf.costd); 
#else
    vec3 F = F0 + (1.0 - F0) * pow5(1.0 - brdf.costd); 
#endif    
    return F;    
}
/*
vec3 getDirectColor(Brdf brdf, Surface surface) {        
    float frk = 0.5 + 2.0 * brdf.costd * brdf.costd * surface.roughness;        
    vec3 rgb =  surface.color * ONE_OVER_PI * (1.0 + (frk - 1.0) * pow5(1.0 - brdf.costl)) * (1.0 + (frk - 1.0) * pow5(1.0 - brdf.costv));
    // vec3 rgb = surface.color * ONE_OVER_PI; // lambert
    return rgb;
}
*/

Brdf getBrdf(Surface surface, vec3 direction) {
    vec3 viewDir = normalize(global.camera.position - surface.point);
    vec3 halfDir = normalize(direction + viewDir);
    float costh = dot(surface.normal, halfDir); 
    float costd = dot(direction, halfDir);      
    float costl = dot(surface.normal, direction);
    float costv = dot(surface.normal, viewDir);
    return Brdf(viewDir, halfDir, costh, costd, costl, costv);
}

vec3 getEnvironmentLight(Surface surface, vec3 direction, vec3 color) {
    Brdf brdf = getBrdf(surface, direction);
    float ndl = clamp(brdf.costl, 0.0, 1.0);    
    vec3 rgb = vec3(0.0);
    if (ndl > 0.0) {
        // float distribution = getDistributionStrength(brdf, surface.roughness);
        float geometry = getGeometryStrength(brdf, surface.roughness);
        vec3 fresnel = getFresnelColor(brdf, surface);
        // Combines the BRDF as well as the pdf of this particular
        // sample direction.
        vec3 specular = color * geometry * fresnel * brdf.costd / (brdf.costh * brdf.costv);       
        float shadow = 1.0;
        if (flags.shadows) { 
            shadow = min(1.0 - surface.selfShadow, getSoftShadowStrength(surface.point, direction, 0.02, 20.0, 7.));        
        }
        rgb = specular * shadow * color;
    }
    return rgb;
}

vec3 getEnvironmentColor(Surface surface, vec3 tint) {
    vec3 viewDir = normalize(surface.point - global.camera.position);    
    vec3 envDir = reflect(viewDir, surface.normal);
    // This is pretty hacky for a microfacet model. We are only
    // sampling the environment in one direction when we should be
    // using many samples and weight them based on their distribution.
    // So to compensate for the hack, I blend towards the blurred version
    // of the cube map as roughness goes up and decrease the light
    // contribution as roughness goes up.
    vec4 specular = .4 * mix(texture2D(iChannel0, envDir.xy), texture2D(iChannel1, envDir.xy), surface.roughness) * (1.0 - surface.roughness);    
    vec3 rgb = getEnvironmentLight(surface, envDir, tint * specular.rgb);
    return rgb;
}

vec3 getLightColor(Surface surface, vec3 direction, vec3 color) {
    Brdf brdf = getBrdf(surface, direction);
    vec3 rgb = vec3(0.0);
    float costl = clamp(brdf.costl, 0.0, 1.0);
    if (costl > 0.0) {
        // remap hotness of roughness for analytic lights
        float roughness = max(0.05, surface.roughness);
        float distribution = getDistributionStrength(brdf, roughness);
        float geometry = getGeometryStrength(brdf, (roughness + 1.0) * 0.5);
        vec3 fresnel = getFresnelColor(brdf, surface);
        vec3 specular = distribution * fresnel * geometry / (4. * brdf.costl * brdf.costv);        
        float shadow = 1.0;
        if (flags.shadows) {
            shadow = min(1.0 - surface.selfShadow, getSoftShadowStrength(surface.point, direction, 0.1, 20.0, 5.));
        }
        // vec3 diff = getDirectColor(brdf, surface);
        // rgb += diff * costl * shadow * color;
        rgb += specular * costl * shadow * color;
    }
    return rgb;
}

vec3 getSurfaceColor(Surface surface) {    
    vec3 ambient = surface.color * .02;
    // ambient occlusion is amount of occlusion.  So 1 is fully occluded
    // and 0 is not occluded at all.  Makes math easier when mixing 
    // shadowing effects.
    float occlusion = 0.0;
    if (flags.occlusions) {
        occlusion = getOcclusionStrength(surface.point, surface.normal);
    }
    vec3 rgb = getLightColor(surface, global.light.direction, global.light.color);
    if (flags.environment) {
        rgb += getEnvironmentColor(surface, global.light.color) * (1.0 - 3.5 * occlusion);
    }
    rgb += ambient * (1.0 - 3.5 * occlusion);
    return rgb;
}

// SCENE
#define GET_SURFACE(p, n) Surface(p, n, vec3(0.0), 0.0, 1.0, 0.0, 0.0)
vec3 getNormal(vec3 p) {
    vec3 epsilon = vec3(0.001, 0.0, 0.0);
    vec3 z = vec3(0.0);
    vec3 n = vec3(
        scenedf(p + epsilon.xyy, z).x - scenedf(p - epsilon.xyy, z).x,
        scenedf(p + epsilon.yxy, z).x - scenedf(p - epsilon.yxy, z).x,
        scenedf(p + epsilon.yyx, z).x - scenedf(p - epsilon.yyx, z).x
    );
    return normalize(n);
}
// ANIMATION
mat3 rotationAroundYAxis(float c, float s) {
    return mat3(c, 0.0, s , 0.0, 1.0, 0.0, -s, 0.0, c);
}
mat3 rotationAroundXAxis(float c, float s) {
    return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
}
void animate() {
    // remap the mouse click ([-1, 1], [-1, 1])
    vec2 click = iMouse.xy / iResolution.xy;    
    // if click isn't initialized (negative), have reasonable defaults
    click = -1.0 + click * 2.0;
    global.time = iGlobalTime;
    // camera position
    global.camera.position = vec3(-13.0, 6.0, 0.0);    
    float roty    = PI * click.x + global.time * 0.1;
    float cosroty = cos(roty);
    float sinroty = sin(roty);
    float rotx    = PI * 0.4 * (0.5 + click.y * 0.5);
    float cosrotx = cos(rotx);
    float sinrotx = sin(rotx);
    // Rotate the camera around the origin
    global.camera.position = rotationAroundYAxis(cosroty, sinroty) * rotationAroundXAxis(cosrotx, sinrotx) * global.camera.position;    
    global.camera.target   = vec3(0.0, -1.0, 0.0);    
    float lroty    = global.time * 0.9;
    float coslroty = cos(lroty);
    float sinlroty = sin(lroty);
    // Rotate the light around the origin
    global.light.direction = rotationAroundYAxis(coslroty, sinlroty) * global.light.direction;
    global.light.color = vec3(1.0);	
}
// CAMERA
CameraData getCamera(vec4 fragCoord) {
    // aspect ratio
    float invar = iResolution.y / iResolution.x;
    vec2 st = fragCoord.xy / iResolution.xy - 0.5;
    st.y *= invar;
    // calculate the ray origin and ray direction that represents
    // mapping the image plane towards the scene
    vec3 iu = vec3(0.0, 1.0, 0.0);
    vec3 iz = normalize(global.camera.target - global.camera.position);
    vec3 ix = normalize(cross(iz, iu));
    vec3 iy = cross(ix, iz);
    vec3 direction = normalize(st.x * ix + st.y * iy + 1.0 * iz);
    return CameraData(global.camera.position, direction, st);
}

void main() {   
    vec3 rgb = vec3(0.0);
    // SCENE
    animate();
    CameraData camera = getCamera(gl_FragCoord);
    vec3 march = getDistanceMarch(camera.position, camera.direction, DM_MAX);
    if (march.y > 0.0) {
        vec3 point = camera.position + march.x * camera.direction;
        vec3 normal = getNormal(point);
        Surface surface = GET_SURFACE(point, normal);
        setMaterial(march.y, march.z, surface);
        // SHADING
        rgb = getSurfaceColor(surface);
    }
    // POST PROCESSING
    if (flags.post) {
        // fall off exponentially into the distance (as if there is a spot light on the point of interest).
        rgb *= exp(-0.003 * (march.x * march.x - 20.0 * 20.0));
        // Gamma correct
        rgb = pow(rgb, vec3(0.45));
        // Contrast adjust - cute trick learned from iq
        rgb = mix(rgb, vec3(dot(rgb, vec3(0.333))), -0.6);
        // color tint
        rgb = 0.5 * rgb + 0.5 * rgb * vec3(1.0, 1.0, 0.9);
    }
    if (global.debug.a > 0.0) {
        gl_FragColor.rgb = global.debug.rgb;
    } else {
        gl_FragColor.rgb = rgb;
    }
    gl_FragColor.a = 1.0;
}
