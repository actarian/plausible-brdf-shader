
/*
// directLight is an out parameter as having it as a return value caused compiler errors on some devices
void getPointDirectLightIrradiance(const in PointLight pointLight, const in GeometricContext geometry, out IncidentLight directLight) {
	vec3 lVector = pointLight.position - geometry.position;
	directLight.direction = normalize(lVector);
	float lightDistance = length(lVector);
	directLight.color = pointLight.color;
	directLight.color *= punctualLightIntensityToIrradianceFactor(lightDistance, pointLight.distance, pointLight.decay);
	directLight.visible = (directLight.color != vec3( 0.0 ));
}
*/

/*
void getPointDirectLightIrradiance( const in PointLight pointLight, const in GeometricContext geometry, out IncidentLight directLight ) {
	vec3 lVector = pointLight.position - geometry.position;
	directLight.direction = normalize( lVector );
	float lightDistance = length( lVector );
	directLight.color = pointLight.color;
	directLight.color *= punctualLightIntensityToIrradianceFactor( lightDistance, pointLight.distance, pointLight.decay );
	directLight.visible = ( directLight.color != vec3( 0.0 ) );
}
*/

/*
float punctualLightIntensityToIrradianceFactor(const in float lightDistance, const in float cutoffDistance, const in float decayExponent) {
		if( decayExponent > 0.0 ) {
#if defined ( PHYSICALLY_CORRECT_LIGHTS )
			float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
			float maxDistanceCutoffFactor = pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
			return distanceFalloff * maxDistanceCutoffFactor;
#else
			return pow( saturate( -lightDistance / cutoffDistance + 1.0 ), decayExponent );
#endif
		}
		return 1.0;
}
*/

/*
vec3 BRDF_Diffuse_Lambert(const in vec3 diffuseColor) {
	return ONE_OVER_PI * diffuseColor;
}
*/

/*
vec3 F_Schlick( const in vec3 specularColor, const in float dotLH ) {
	float fresnel = exp2( ( -5.55473 * dotLH - 6.98316 ) * dotLH );
	return ( 1.0 - specularColor ) * fresnel + specularColor;
}
float G_GGX_Smith( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gl = dotNL + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	float gv = dotNV + sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	return 1.0 / ( gl * gv );
}
float G_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON_B );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0; // avoid alpha = 0 with dotNH = 1
	return ONE_OVER_PI * a2 / pow2( denom );
}
vec3 BRDF_Specular_GGX( const in IncidentLight incidentLight, const in GeometricContext geometry, const in vec3 specularColor, const in float roughness ) {
	float alpha = pow2( roughness ); // UE4's roughness
	vec3 halfDir = normalize( incidentLight.direction + geometry.viewDir );
	float dotNL = saturate( dot( geometry.normal, incidentLight.direction ) );
	float dotNV = saturate( dot( geometry.normal, geometry.viewDir ) );
	float dotNH = saturate( dot( geometry.normal, halfDir ) );
	float dotLH = saturate( dot( incidentLight.direction, halfDir ) );
	vec3 F = F_Schlick( specularColor, dotLH );
	float G = G_GGX_SmithCorrelated( alpha, dotNL, dotNV );
	float D = D_GGX( alpha, dotNH );
	return F * ( G * D );
}
*/
/*
vec4 GammaToLinear(in vec4 value, in float gammaFactor) {
  return vec4(pow(value.xyz, vec3(gammaFactor)), value.w);
}
vec4 mapTexelToLinear(vec4 value) { 
	return GammaToLinear( value, float(GAMMA_FACTOR)); 
}
*/

