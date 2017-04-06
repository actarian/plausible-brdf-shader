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

// GLOBALS
vec3  g_camPointAt   = vec3(0.0);
vec3  g_camOrigin    = vec3(0.0);
float g_time         = 0.0;
vec3  g_ldir         = vec3(0.5, 0.5, 0.0);
float g_lillum       = 1.0;
vec4  g_debugcolor   = vec4(0.0);

// UTILITIES
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

mat3 rotMatAroundYAxis(float c, float s) {
    return mat3(c, 0.0, s , 0.0, 1.0, 0.0, -s, 0.0, c);
}

mat3 rotMatAroundXAxis(float c, float s) {
    return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
}

float pow5(float v){
    float tmp = v * v;
    return tmp * tmp * v;
}

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

// Returns a vec2 where:
//   result.x = 1.0 or 0.0 if there is a hit
//   result.y = t such that origin + t * dir = hit point
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
        vec2 rx = intersectDSPlane(ro, rd, vec3( -srd.x, 0.0, 0.0), vec3(0.5 * bounds.x *  -srd.x, 0.0, 0.0));
        if (rx.x > 0.5 && inbounds(ro + rd * (rx.y + EPSILON), bounds)) {
            d = min(d, rx.y);
        }
        vec2 ry = intersectDSPlane(ro, rd, vec3(0.0, -srd.y, 0.0), vec3(0.0, 0.5 * bounds.y *  -srd.y, 0.0));
        if (ry.x > 0.5 && inbounds(ro + rd * (ry.y + EPSILON), bounds)) {
            d = min(d, ry.y);
        }
        vec2 rz = intersectDSPlane(ro, rd, vec3(0.0, 0.0, -srd.z), vec3(0.0, 0.0, 0.5 * bounds.z *  -srd.z));
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
        float aobuffer = 1.0;
        vec3 bounds = rbuffer * vec3(float(ROWS + 1), 1.0, float(COLS + 1));
        // Add a buffer to the bounding box to account for the 
        // ambient occlusion marching.
        bounds  += aobuffer;    
        tbox = intersectBox(pos, rd, bounds);
    }    
    if (tbox < 10.0) {
        vec3 bobj = ballsobj(pos + rd * tbox, r);
        // Add the distance to the bounding box
        bobj.x  += tbox;
        obj = mergeobjs(obj, bobj);
    }    
    // vec3 bobj = ballsobj(pos, r);
    // obj = mergeobjs(obj, bobj);
    obj = mergeobjs(obj, planeobj(pos, 0.5));
    return obj;
}

// DIST MARCH
#define DISTMARCH_STEPS 60
#define DISTMARCH_MAXDIST 50.0
vec3 distmarch(vec3 ro, vec3 rd, float maxd) {
    float dist = 0.01;
    vec3 res = vec3(0.0, -1.0, -1.0);
    for (int i=0; i < DISTMARCH_STEPS; i++) {
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

// SHADOWING & NORMALS
#define SOFTSHADOW_STEPS 40
#define SOFTSHADOW_STEPSIZE 0.1
float calcSoftShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float shadow = 1.0;
    float t = mint;
    for (int i=0; i < SOFTSHADOW_STEPS; i++) {
        if (t < maxt) {
            float h = scenedf(ro + rd * t, rd).x;
            shadow = min(shadow, k * h / t);
            t  += SOFTSHADOW_STEPSIZE;
        }
    }
    return clamp(shadow, 0.00, 1.0);
}

// AMBIENT OCCLUSION
#define AO_NUMSAMPLES 5
#define AO_STEPSIZE 0.14
#define AO_STEPSCALE 0.35
float calcAO(vec3 p, vec3 n) {
    float ao = 0.0;
    float aoscale = 1.0;
    for (int aoi=0; aoi< AO_NUMSAMPLES ; aoi++) {
        float step = 0.01 + AO_STEPSIZE * float(aoi);
        vec3 aop =  n * step + p;        
        float d = scenedf(aop, n).x;
        ao += -(d - step) * aoscale;
        aoscale *= AO_STEPSCALE;
    }    
    return clamp(ao, 0.0, 1.0);
}

// SHADING
struct SurfaceData {
    vec3 point;
    vec3 normal;
    vec3 basecolor;
    float roughness;
    float specular;
    float metallic;
    float selfshad;
};

struct BRDFVars {
    // vdir is the view direction vector
    vec3 vdir;
    // The half vector of a microfacet model 
    vec3 hdir;
    // cos(theta_h) - theta_h is angle between half vector and normal
    float costh; 
    // cos(theta_d) - theta_d is angle between half vector and light dir / view dir
    float costd;      
    // cos(theta_l) - theta_l is angle between the light vector and normal
    float costl;
    // cos(theta_v) - theta_v is angle between the viewing vector and normal
    float costv;
};

BRDFVars calcBRDFVars(SurfaceData surf, vec3 ldir) {
    vec3 vdir = normalize(g_camOrigin - surf.point);
    vec3 hdir = normalize(ldir + vdir);
    float costh = dot(surf.normal, hdir); 
    float costd = dot(ldir, hdir);      
    float costl = dot(surf.normal, ldir);
    float costv = dot(surf.normal, vdir);
    return BRDFVars(vdir, hdir, costh, costd, costl, costv);
}

vec3 calcNormal(vec3 p) {
    vec3 epsilon = vec3(0.001, 0.0, 0.0);
    vec3 z = vec3(0.0);
    vec3 n = vec3(
        scenedf(p + epsilon.xyy, z).x - scenedf(p - epsilon.xyy, z).x,
        scenedf(p + epsilon.yxy, z).x - scenedf(p - epsilon.yxy, z).x,
        scenedf(p + epsilon.yyx, z).x - scenedf(p - epsilon.yyx, z).x
    );
    return normalize(n);
}

void material(float matid, float surfid, inout SurfaceData surf) {
    vec3 surfcol = vec3(1.0);
    if (matid - 0.5 < SPHERE_MATL) { 
        surf.basecolor = vec3(1.0, 0.35, 0.5); 
        float ballrow = floor(surfid / float(COLS));
        surf.roughness = mod(surfid, float(COLS)) / float(COLS - 1);
        surf.metallic = floor(surfid / float(COLS)) / float(ROWS - 1);
        surf.specular = 0.8;
    } else if (matid - 0.5 < PLANE_MATL) {
        vec4 pavem = texture2D(iChannel2, 0.1 * surf.point.xz);
        surf.basecolor = vec3(.1 * smoothstep(0.6, 0.3, pavem.r));
        surf.metallic = 0.0;
        surf.roughness = 0.6;
        surf.specular = 0.02; // .05        
        surf.normal.xz  += 0.1 * pavem.bg;
        surf.normal = normalize(surf.normal);
        surf.selfshad = 0.3 * (1.0 - smoothstep(0.4, 0.8, pavem.g));
    }
}

vec3 calcDiff(BRDFVars bvars, SurfaceData surf) {        
    float frk = 0.5 + 2.0 *  bvars.costd * bvars.costd * surf.roughness;        
    return surf.basecolor * ONE_OVER_PI * (1.0 + (frk - 1.0) * pow5(1.0 - bvars.costl)) * (1.0 + (frk - 1.0) * pow5(1.0 - bvars.costv));
    // return surf.basecolor * ONE_OVER_PI; // lambert
}

float calcDistrScalar(BRDFVars bvars, float roughness) {
    // D(h) factor
    // using the GGX approximation where the gamma factor is 2.0
    // Clamping roughness so that a directional light has a specular
    // response.  A roughness of perfectly 0 will create light 
    // singularities.
    float alpha = roughness * roughness;
    float denom = bvars.costh * bvars.costh * (alpha * alpha - 1.0) + 1.0;
    float D = (alpha * alpha) / (PI * denom * denom); 
    // using the GTR approximation where the gamma factor is generalized
    // float gamma = 1.0;
    // float sinth = length(cross(surf.normal, bvars.hdir));
    // float D = 1.0 / pow(alpha * alpha * bvars.costh * bvars.costh + sinth * sinth, gamma);
    return D;
}

float calcGeomScalar(BRDFVars bvars, float roughness) {    
    // G(h,l,v) factor    
    float k = roughness / 2.0;
    float Gv = step(0.0, bvars.costv) * (bvars.costv / (bvars.costv * (1.0 - k) + k));
    float Gl = step(0.0, bvars.costl) * (bvars.costl / (bvars.costl * (1.0 - k) + k));
    return Gl * Gv;
}


vec3 calcFresnelColor(BRDFVars bvars, SurfaceData surf) {
    // F(h,l) factor
    vec3 F0 = surf.specular * mix(vec3(1.0), surf.basecolor, surf.metallic);    
#if FRESNEL_HORNER_APPROXIMATION
    vec3 F = F0 + (1.0 - F0) * exp2((-5.55473 * bvars.costd - 6.98316) * bvars.costd); 
#else
    vec3 F = F0 + (1.0 - F0) * pow5(1.0 - bvars.costd); 
#endif    
    return F;    
}

vec3 integrateDirLight(vec3 ldir, vec3 lcolor, SurfaceData surf) {
    BRDFVars bvars = calcBRDFVars(surf, ldir);
    vec3 cout = vec3(0.0);
    float ndl = clamp(bvars.costl, 0.0, 1.0);
    if (ndl > 0.0) {
        vec3 diff = calcDiff(bvars, surf);
        // remap hotness of roughness for analytic lights
        float rroughness = max(0.05, surf.roughness);
        float D = calcDistrScalar(bvars, rroughness);
        float G = calcGeomScalar(bvars, (rroughness + 1.0) * 0.5);
        vec3 F  = calcFresnelColor(bvars, surf);
        vec3 spec = D * F * G / (4. * bvars.costl * bvars.costv);        
        float shd = min(1.0 - surf.selfshad, calcSoftShadow(surf.point, ldir, 0.1, 20.0, 5.));
        // cout   += diff * ndl * shd * lcolor;
        cout   += spec * ndl * shd * lcolor;
    }
    return cout;
}

vec3 sampleEnvLight(vec3 ldir, vec3 lcolor, SurfaceData surf) {
    BRDFVars bvars = calcBRDFVars(surf, ldir);
    float ndl = clamp(bvars.costl, 0.0, 1.0);    
    vec3 cout = vec3(0.0);
    if (ndl > 0.0) {
        float D = calcDistrScalar(bvars, surf.roughness);
        float G = calcGeomScalar(bvars, surf.roughness);
        vec3 F  = calcFresnelColor(bvars, surf);
        // Combines the BRDF as well as the pdf of this particular
        // sample direction.
        vec3 spec = lcolor * G * F * bvars.costd / (bvars.costh * bvars.costv);        
        float shd = min(1.0 - surf.selfshad, calcSoftShadow(surf.point, ldir, 0.02, 20.0, 7.));        
        cout = spec * shd * lcolor;
    }
    return cout;
}

vec3 integrateEnvLight(SurfaceData surf, vec3 tint) {
    vec3 vdir = normalize(surf.point - g_camOrigin);    
    vec3 envdir = reflect(vdir, surf.normal);
    // This is pretty hacky for a microfacet model. We are only
    // sampling the environment in one direction when we should be
    // using many samples and weight them based on their distribution.
    // So to compensate for the hack, I blend towards the blurred version
    // of the cube map as roughness goes up and decrease the light
    // contribution as roughness goes up.
    vec4 specolor = .4 * mix(texture2D(iChannel0, envdir.xy), texture2D(iChannel1, envdir.xy), surf.roughness) * (1.0 - surf.roughness);    
    vec3 envspec = sampleEnvLight(envdir, tint * specolor.rgb, surf);
    return envspec;
}

vec3 shadeSurface(SurfaceData surf) {    
    vec3 amb = surf.basecolor * .02;
    // ambient occlusion is amount of occlusion.  So 1 is fully occluded
    // and 0 is not occluded at all.  Makes math easier when mixing 
    // shadowing effects.
    float ao = calcAO(surf.point, surf.normal);
    vec3 cout = integrateDirLight(g_ldir, vec3(g_lillum), surf);
    cout += integrateEnvLight(surf, vec3(g_lillum)) * (1.0 - 3.5 * ao);
    cout += amb * (1.0 - 3.5 * ao);
    return cout;
}

// CAMERA & GLOBALS
void animateGlobals() {
    // remap the mouse click ([-1, 1], [-1, 1])
    vec2 click = iMouse.xy / iResolution.xy;    
    // if click isn't initialized (negative), have reasonable defaults
    click = -1.0 + click * 2.0;
    g_time = iGlobalTime;
    // camera position
    g_camOrigin = vec3(-13.0, 6.0, 0.0);    
    float roty    = PI * click.x + g_time * 0.1;
    float cosroty = cos(roty);
    float sinroty = sin(roty);
    float rotx    = PI * 0.4 * (0.5 + click.y * 0.5);
    float cosrotx = cos(rotx);
    float sinrotx = sin(rotx);
    g_lillum = .8;	
    // Rotate the camera around the origin
    g_camOrigin = rotMatAroundYAxis(cosroty, sinroty) * rotMatAroundXAxis(cosrotx, sinrotx) * g_camOrigin;    
    g_camPointAt   = vec3(0.0, -1.0, 0.0);    
    float lroty    = g_time * 0.9;
    float coslroty = cos(lroty);
    float sinlroty = sin(lroty);
    // Rotate the light around the origin
    g_ldir = rotMatAroundYAxis(coslroty, sinlroty) * g_ldir;
}

// CAMERA
struct CameraData {
    vec3 origin;
    vec3 dir;
    vec2 st;
};

CameraData setupCamera(vec4 fragCoord) {
    // aspect ratio
    float invar = iResolution.y / iResolution.x;
    vec2 st = fragCoord.xy / iResolution.xy - 0.5;
    st.y *= invar;
    // calculate the ray origin and ray direction that represents
    // mapping the image plane towards the scene
    vec3 iu = vec3(0.0, 1.0, 0.0);
    vec3 iz = normalize(g_camPointAt - g_camOrigin);
    vec3 ix = normalize(cross(iz, iu));
    vec3 iy = cross(ix, iz);
    vec3 dir = normalize(st.x * ix + st.y * iy + 1.0 * iz);
    return CameraData(g_camOrigin, dir, st);
}

#define INITSURF(p, n) SurfaceData(p, n, vec3(0.0), 0.0, 1.0, 0.0, 0.0)
void main() {   
    animateGlobals();
    CameraData cam = setupCamera(gl_FragCoord);
    vec3 scenemarch = distmarch(cam.origin, cam.dir, DISTMARCH_MAXDIST);
    // SHADING
    vec3 scenecol = vec3(0.0);
    if (scenemarch.y > 0.0) {
        vec3 mp = cam.origin + scenemarch.x * cam.dir;
        vec3 mn = calcNormal(mp);
        SurfaceData currSurf = INITSURF(mp, mn);
        material(scenemarch.y, scenemarch.z, currSurf);
        scenecol = shadeSurface(currSurf);
    }
    // POST PROCESSING
    // fall off exponentially into the distance (as if there is a spot light
    // on the point of interest).
    scenecol  *= exp(-0.003 * (scenemarch.x * scenemarch.x - 20.0 * 20.0));
    // Gamma correct
    scenecol = pow(scenecol, vec3(0.45));
    // Contrast adjust - cute trick learned from iq
    scenecol = mix(scenecol, vec3(dot(scenecol,vec3(0.333))), -0.6);
    // color tint
    scenecol = 0.5 * scenecol + 0.5 * scenecol * vec3(1.0, 1.0, 0.9);
    if (g_debugcolor.a > 0.0) {
        gl_FragColor.rgb = g_debugcolor.rgb;
    } else {
        gl_FragColor.rgb = scenecol;
    }
    gl_FragColor.a = 1.0;
}
