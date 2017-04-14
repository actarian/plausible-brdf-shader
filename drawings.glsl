// Author:
// Title:

#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform vec2 u_mouse;
uniform float u_time;

vec2 pixel = vec2(1.0) / u_resolution;

float drawLine(vec2 p, float y, float size) {
	return smoothstep(y - pixel.x * size, y, p.y) - smoothstep(y, y + pixel.y * size, p.y);
}
float drawLine(vec2 p, float y) {
	return drawLine(p, y, 1.0);
}

void main() {
    vec2 st = gl_FragCoord.xy / u_resolution.xy;
    st.x *= u_resolution.x / u_resolution.y;
    
	vec2 uv = -1.0 + st * 2.0;        
    
    vec3 rgb = vec3(0.0);
    
    float y = st.x;
    // float y = step(0.1, st.x);
    // float y = pow(st.x, 10.0);
    // float x = st.x * 20.0;
    // float y = sin(x) + sin(x / 0.5) * 0.5 + sin(x / 0.333) * 0.333 + sin(x / 0.25) * 0.25 + sin(x / 0.2) * 0.2;
    
    float line = drawLine(st, y);
    rgb += line;
    
    gl_FragColor = vec4(rgb, 1.0);
}