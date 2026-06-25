/*
 * Simulacion N-cuerpos: Galaxia 3D
 * Computacion Paralela - UTEM
 *
 * Compilar:
 *   nvcc main.cu -o galaxy -lGL -lGLU -lglfw -lGLEW -O3 -arch=sm_75
 *
 * Dependencias:
 *   sudo apt install libglfw3-dev libglew-dev
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// OpenGL / GLFW / GLEW
#include <GL/glew.h>
#include <GLFW/glfw3.h>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// ─── Parametros de simulacion ──────────────────────────────────────────────
#define N_DEFAULT       50000       // numero de cuerpos por defecto
#define TILE_SIZE       256         // threads por bloque (multiplo de 32)
#define G               0.00015f    // constante gravitacional escalada
#define EPSILON2        0.08f       // suavizador (evita singularidades)
#define DT              0.004f      // paso de tiempo
#define MASS_CENTRAL    8000.0f     // masa del nucleo galactico
#define NUM_ARMS        4           // brazos espirales

// ─── Ventana ───────────────────────────────────────────────────────────────
#define WIN_W           1280
#define WIN_H           720

// ─── Macros de error ───────────────────────────────────────────────────────
#define CUDA_CHECK(call) do {                                       \
    cudaError_t e = (call);                                         \
    if (e != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(e));         \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

#define GL_CHECK() do {                                             \
    GLenum err = glGetError();                                      \
    if (err != GL_NO_ERROR)                                         \
        fprintf(stderr, "GL error %s:%d: 0x%x\n",                  \
                __FILE__, __LINE__, err);                           \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════
//  ESTRUCTURAS
// ═══════════════════════════════════════════════════════════════════════════

// float4: x,y,z = posicion/velocidad,  w = masa (en pos) / 0 (en vel)
typedef float4 Body;

// Estado de camara (controlada por mouse)
typedef struct {
    float rotX, rotY;       // angulos de orbita
    float zoom;             // distancia
    double lastMouseX, lastMouseY;
    int dragging;
} Camera;

// Estado global de la aplicacion
typedef struct {
    int    N;               // numero de cuerpos actual
    int    paused;
    int    showHelp;
    double simTime;         // tiempo acumulado de simulacion
    long   steps;           // pasos ejecutados
    float  fps;
    double lastFPSTime;
    int    fpsFrames;
} AppState;

// ═══════════════════════════════════════════════════════════════════════════
//  KERNELS CUDA
// ═══════════════════════════════════════════════════════════════════════════

/*
 * Kernel de fuerzas gravitacionales con Shared Memory Tiling.
 *
 * Cada bloque carga TILE_SIZE cuerpos a shared memory en cada iteracion
 * del loop externo. Todos los threads del bloque calculan la fuerza de
 * interaccion contra esos TILE_SIZE cuerpos. Esto reduce accesos a VRAM
 * de O(N^2) a O(N^2 / TILE_SIZE).
 *
 * pos[i] = (x, y, z, masa)
 * acc[i] = aceleracion acumulada para el cuerpo i
 */
__global__ void kernelFuerzas(const Body* __restrict__ pos,
                               Body*       __restrict__ acc,
                               int N)
{
    extern __shared__ Body tile[];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    float3 pi  = {0,0,0};
    float3 ai  = {0,0,0};

    if (i < N) {
        pi.x = pos[i].x;
        pi.y = pos[i].y;
        pi.z = pos[i].z;
    }

    // Loop sobre tiles de TILE_SIZE cuerpos
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Carga cooperativa: cada thread carga UN cuerpo al tile
        int jGlobal = t * TILE_SIZE + threadIdx.x;
        tile[threadIdx.x] = (jGlobal < N) ? pos[jGlobal] : make_float4(0,0,0,0);
        __syncthreads();

        // Calcula fuerza contra los TILE_SIZE cuerpos del tile
        #pragma unroll 8
        for (int j = 0; j < TILE_SIZE; j++) {
            float rx = tile[j].x - pi.x;
            float ry = tile[j].y - pi.y;
            float rz = tile[j].z - pi.z;
            float dist2 = rx*rx + ry*ry + rz*rz + EPSILON2;
            // rsqrtf: instruccion nativa GPU, ~4 ciclos vs ~20 de sqrtf+div
            float inv3  = rsqrtf(dist2 * dist2 * dist2);
            float fmag  = G * tile[j].w * inv3;
            ai.x += fmag * rx;
            ai.y += fmag * ry;
            ai.z += fmag * rz;
        }
        __syncthreads();
    }

    if (i < N)
        acc[i] = make_float4(ai.x, ai.y, ai.z, 0.0f);
}

/*
 * Kernel de integracion Leapfrog.
 *
 * Leapfrog es un integrador simplectico: conserva la energia mecanica
 * mucho mejor que Euler explicito para orbitas a largo plazo.
 *
 * v(t+dt/2) = v(t-dt/2) + a(t)*dt
 * x(t+dt)   = x(t) + v(t+dt/2)*dt
 *
 * pos[i].w = masa (no se modifica)
 * vel[i].w = 0 (no usado)
 */
__global__ void kernelLeapfrog(Body* __restrict__ pos,
                                Body* __restrict__ vel,
                                const Body* __restrict__ acc,
                                int N, float dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Actualiza velocidad (medio paso adelante)
    float vx = vel[i].x + acc[i].x * dt;
    float vy = vel[i].y + acc[i].y * dt;
    float vz = vel[i].z + acc[i].z * dt;

    // Actualiza posicion
    float px = pos[i].x + vx * dt;
    float py = pos[i].y + vy * dt;
    float pz = pos[i].z + vz * dt;

    vel[i] = make_float4(vx, vy, vz, 0.0f);
    pos[i] = make_float4(px, py, pz, pos[i].w);  // conserva masa
}

/*
 * Kernel auxiliar: copia posiciones al VBO de OpenGL para render.
 * devVBO apunta directamente al buffer de la GPU compartido con OpenGL.
 * Sin cudaMemcpy, sin round-trip CPU.
 */
__global__ void kernelCopiaVBO(const Body* __restrict__ pos,
                                float*      __restrict__ devVBO,
                                int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    // VBO layout: x,y,z,speed (speed = longitud de velocidad, para colorear)
    devVBO[i*4+0] = pos[i].x;
    devVBO[i*4+1] = pos[i].y;
    devVBO[i*4+2] = pos[i].z;
    devVBO[i*4+3] = pos[i].w;  // masa → usada para tamaño del punto
}

// ═══════════════════════════════════════════════════════════════════════════
//  INICIALIZACION DE LA GALAXIA
// ═══════════════════════════════════════════════════════════════════════════

static float randf() { return (float)rand() / (float)RAND_MAX; }

/*
 * Genera una galaxia espiral con:
 *   - NUM_ARMS brazos logaritmicos
 *   - Distribucion radial tipo Plummer (sqrt para densidad decreciente)
 *   - Velocidades tangenciales keplerianamente correctas
 *   - Agujero negro central de masa MASS_CENTRAL
 *   - Disco delgado: dispersion vertical proporcional a exp(-r)
 */
void initGalaxy(Body* hPos, Body* hVel, int N)
{
    srand((unsigned)time(NULL));
    const float PI2 = 2.0f * 3.14159265f;

    // Agujero negro central (primeros 3 slots, inmovil)
    for (int k = 0; k < 3 && k < N; k++) {
        hPos[k] = make_float4(0.0f, 0.0f, 0.0f, MASS_CENTRAL / 3.0f);
        hVel[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Estrellas del disco
    for (int i = 3; i < N; i++) {
        int arm = i % NUM_ARMS;

        // Radio: distribucion de Plummer aplastada hacia el centro
        float r = sqrtf(randf()) * 12.0f + 0.3f;

        // Angulo del brazo espiral (logaritmico)
        float baseAngle = ((float)arm / NUM_ARMS) * PI2 + r * 0.6f;
        float spread    = 0.35f / fmaxf(r * 0.4f, 0.5f);
        float angle     = baseAngle + (randf() - 0.5f) * spread;

        float x = cosf(angle) * r;
        float z = sinf(angle) * r;
        float y = (randf() - 0.5f) * 0.4f * expf(-r * 0.35f);

        // Masa estelar aleatoria (unidades arbitrarias)
        float mass = 0.05f + randf() * 0.3f;

        // Velocidad circular kepleriana: v_c = sqrt(G*M_enc / r)
        // Masa encerrada aproximada (nucleo + disco)
        float Menc = MASS_CENTRAL + (float)(i - 3) * 0.15f;
        float vc   = sqrtf(G * Menc / r) * 0.92f;

        // Perturbaciones termicas pequenas
        float dvx = (randf() - 0.5f) * 0.012f;
        float dvy = (randf() - 0.5f) * 0.005f;
        float dvz = (randf() - 0.5f) * 0.012f;

        hPos[i] = make_float4(x, y, z, mass);
        hVel[i] = make_float4(-sinf(angle)*vc + dvx,
                               dvy,
                               cosf(angle)*vc + dvz,
                               0.0f);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  SHADERS OPENGL (GLSL inline)
// ═══════════════════════════════════════════════════════════════════════════

static const char* VS_SOURCE = R"glsl(
#version 330 core

layout(location = 0) in vec3 inPos;    // x, y, z
layout(location = 1) in float inMass;  // masa -> tamaño del punto

uniform mat4 uMVP;
uniform float uMaxMass;

out float vMass;
out float vDist;  // distancia al centro (para colorear)

void main() {
    gl_Position  = uMVP * vec4(inPos, 1.0);
    float depth  = gl_Position.z / gl_Position.w;
    // Punto mas grande cuanto mayor la masa y mas cerca este
    float sz     = mix(1.5, 5.0, inMass / uMaxMass);
    gl_PointSize = sz * (1.0 - depth * 0.3);
    vMass        = inMass;
    vDist        = length(inPos.xz);  // radio en el plano del disco
}
)glsl";

static const char* FS_SOURCE = R"glsl(
#version 330 core

in float vMass;
in float vDist;
out vec4 fragColor;

uniform float uMaxDist;

void main() {
    // Forma circular suave del punto
    vec2  uv = gl_PointCoord - 0.5;
    float d  = length(uv) * 2.0;
    float a  = 1.0 - smoothstep(0.5, 1.0, d);
    if (a < 0.01) discard;

    // Color segun radio orbital:
    //   centro (r~0) -> blanco/amarillo caliente
    //   medio        -> azul-blanco
    //   borde        -> azul frio
    float t = clamp(vDist / uMaxDist, 0.0, 1.0);

    vec3 colorCore  = vec3(1.0,  0.95, 0.7);   // blanco calido
    vec3 colorMid   = vec3(0.55, 0.75, 1.0);   // azul claro
    vec3 colorEdge  = vec3(0.2,  0.35, 0.9);   // azul frio

    vec3 c = mix(colorCore, colorMid,  clamp(t * 2.0, 0.0, 1.0));
    c      = mix(c,         colorEdge, clamp((t - 0.5) * 2.0, 0.0, 1.0));

    // Nucleo galactico: mucho mas brillante
    if (vDist < 0.5) {
        c = mix(vec3(1.0, 0.98, 0.9), c, vDist / 0.5);
        a = min(a * 2.5, 1.0);
    }

    fragColor = vec4(c * a, a * 0.88);
}
)glsl";

// ═══════════════════════════════════════════════════════════════════════════
//  OPENGL: compilar shaders, matrices
// ═══════════════════════════════════════════════════════════════════════════

static GLuint compileShader(GLenum type, const char* src)
{
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, NULL);
    glCompileShader(s);
    GLint ok; glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetShaderInfoLog(s, sizeof(buf), NULL, buf);
        fprintf(stderr, "Shader error:\n%s\n", buf);
    }
    return s;
}

static GLuint buildProgram(const char* vs, const char* fs)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, compileShader(GL_VERTEX_SHADER,   vs));
    glAttachShader(prog, compileShader(GL_FRAGMENT_SHADER, fs));
    glLinkProgram(prog);
    GLint ok; glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char buf[1024]; glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
        fprintf(stderr, "Program link error:\n%s\n", buf);
    }
    return prog;
}

// Matrices column-major para OpenGL
typedef float Mat4[16];

static void mat4Identity(Mat4 m)
{
    memset(m, 0, sizeof(Mat4));
    m[0]=m[5]=m[10]=m[15]=1.0f;
}

static void mat4Perspective(Mat4 m, float fovY, float aspect, float n, float f)
{
    memset(m, 0, sizeof(Mat4));
    float t = tanf(fovY / 2.0f);
    m[0]  =  1.0f / (aspect * t);
    m[5]  =  1.0f / t;
    m[10] = -(f + n) / (f - n);
    m[11] = -1.0f;
    m[14] = -(2.0f * f * n) / (f - n);
}

static void mat4Mul(Mat4 out, const Mat4 a, const Mat4 b)
{
    Mat4 tmp;
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) {
        tmp[i*4+j] = 0;
        for (int k=0;k<4;k++) tmp[i*4+j] += a[i*4+k]*b[k*4+j];
    }
    memcpy(out, tmp, sizeof(Mat4));
}

static void mat4RotX(Mat4 m, float a)
{
    mat4Identity(m);
    m[5]=cosf(a); m[6]=-sinf(a);
    m[9]=sinf(a); m[10]=cosf(a);
}
static void mat4RotY(Mat4 m, float a)
{
    mat4Identity(m);
    m[0]=cosf(a); m[2]=sinf(a);
    m[8]=-sinf(a); m[10]=cosf(a);
}
static void mat4Trans(Mat4 m, float x, float y, float z)
{
    mat4Identity(m);
    m[12]=x; m[13]=y; m[14]=z;
}

// ═══════════════════════════════════════════════════════════════════════════
//  CALLBACKS GLFW
// ═══════════════════════════════════════════════════════════════════════════

static Camera   g_cam   = {0.4f, 0.0f, 20.0f, 0, 0, 0};
static AppState g_app   = {N_DEFAULT, 0, 1, 0.0, 0, 0.0f, 0.0, 0};

static void cbMouseButton(GLFWwindow* w, int btn, int action, int mods)
{
    (void)mods;
    if (btn == GLFW_MOUSE_BUTTON_LEFT)
        g_cam.dragging = (action == GLFW_PRESS);
    if (action == GLFW_PRESS)
        glfwGetCursorPos(w, &g_cam.lastMouseX, &g_cam.lastMouseY);
}

static void cbMouseMove(GLFWwindow* w, double x, double y)
{
    if (!g_cam.dragging) return;
    float dx = (float)(x - g_cam.lastMouseX) * 0.005f;
    float dy = (float)(y - g_cam.lastMouseY) * 0.005f;
    g_cam.rotY += dx;
    g_cam.rotX += dy;
    g_cam.lastMouseX = x;
    g_cam.lastMouseY = y;
}

static void cbScroll(GLFWwindow* w, double dx, double dy)
{
    (void)w; (void)dx;
    g_cam.zoom = fmaxf(3.0f, fminf(60.0f, g_cam.zoom - (float)dy * 1.5f));
}

static void cbKey(GLFWwindow* w, int key, int sc, int action, int mods)
{
    (void)sc; (void)mods;
    if (action != GLFW_PRESS) return;
    if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q)
        glfwSetWindowShouldClose(w, GLFW_TRUE);
    if (key == GLFW_KEY_SPACE)
        g_app.paused = !g_app.paused;
    if (key == GLFW_KEY_H)
        g_app.showHelp = !g_app.showHelp;
}

static void cbResize(GLFWwindow* w, int width, int height)
{
    (void)w;
    glViewport(0, 0, width, height);
}

// ═══════════════════════════════════════════════════════════════════════════
//  BENCHMARK: comparacion CPU vs GPU
// ═══════════════════════════════════════════════════════════════════════════

void benchmarkCPUvsGPU(int N)
{
    printf("\n=== BENCHMARK CPU vs GPU (N=%d) ===\n", N);

    size_t bytes = N * sizeof(Body);
    Body* hPos = (Body*)malloc(bytes);
    Body* hVel = (Body*)malloc(bytes);
    Body* hAcc = (Body*)malloc(bytes);
    initGalaxy(hPos, hVel, N);

    // --- CPU (un paso de fuerzas) ---
    double t0 = glfwGetTime();
    for (int i = 0; i < N; i++) {
        float ax=0,ay=0,az=0;
        for (int j = 0; j < N; j++) {
            if (i==j) continue;
            float rx = hPos[j].x-hPos[i].x;
            float ry = hPos[j].y-hPos[i].y;
            float rz = hPos[j].z-hPos[i].z;
            float d2 = rx*rx+ry*ry+rz*rz+EPSILON2;
            float inv3 = 1.0f/(d2*sqrtf(d2));
            float f = G*hPos[j].w*inv3;
            ax+=f*rx; ay+=f*ry; az+=f*rz;
        }
        hAcc[i]=make_float4(ax,ay,az,0);
    }
    double cpuMs = (glfwGetTime()-t0)*1000.0;

    // --- GPU ---
    Body *dPos, *dAcc;
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, bytes));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, bytes, cudaMemcpyHostToDevice));

    int grid = (N + TILE_SIZE - 1) / TILE_SIZE;
    size_t shMem = TILE_SIZE * sizeof(Body);

    // warmup
    kernelFuerzas<<<grid,TILE_SIZE,shMem>>>(dPos,dAcc,N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t evStart, evStop;
    cudaEventCreate(&evStart); cudaEventCreate(&evStop);
    cudaEventRecord(evStart);
    kernelFuerzas<<<grid,TILE_SIZE,shMem>>>(dPos,dAcc,N);
    cudaEventRecord(evStop);
    cudaEventSynchronize(evStop);
    float gpuMs=0;
    cudaEventElapsedTime(&gpuMs,evStart,evStop);

    double flops = 20.0 * (double)N * (double)N;
    printf("CPU: %.1f ms\n", cpuMs);
    printf("GPU: %.2f ms\n", gpuMs);
    printf("Speedup: %.1fx\n", cpuMs/gpuMs);
    printf("GPU GFLOPS: %.2f\n", flops / (gpuMs * 1e6));

    cudaFree(dPos); cudaFree(dAcc);
    free(hPos); free(hVel); free(hAcc);
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);
    printf("================================\n\n");
}

// ═══════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════

int main(int argc, char** argv)
{
    int N = N_DEFAULT;
    if (argc > 1) N = atoi(argv[1]);
    if (N < 1024)  N = 1024;
    if (N > 500000) N = 500000;
    g_app.N = N;

    printf("=== Simulacion N-cuerpos: Galaxia 3D ===\n");
    printf("N = %d cuerpos\n", N);
    printf("TILE_SIZE = %d\n", TILE_SIZE);
    printf("Controles: arrastrar=rotar, scroll=zoom, SPACE=pausa, H=ayuda, Q=salir\n\n");

    // ── GLFW / OpenGL ──────────────────────────────────────────────────────
    if (!glfwInit()) { fprintf(stderr, "GLFW init failed\n"); return 1; }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_SAMPLES, 4);

    char title[128];
    snprintf(title, sizeof(title), "Galaxy N-body — N=%d", N);
    GLFWwindow* window = glfwCreateWindow(WIN_W, WIN_H, title, NULL, NULL);
    if (!window) { fprintf(stderr, "Ventana GLFW failed\n"); glfwTerminate(); return 1; }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);  // desactiva VSync para medir FPS real

    glfwSetMouseButtonCallback(window, cbMouseButton);
    glfwSetCursorPosCallback(window, cbMouseMove);
    glfwSetScrollCallback(window, cbScroll);
    glfwSetKeyCallback(window, cbKey);
    glfwSetFramebufferSizeCallback(window, cbResize);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { fprintf(stderr, "GLEW init failed\n"); return 1; }

    // ── Buffers GPU (CUDA) ─────────────────────────────────────────────────
    size_t bytes = N * sizeof(Body);
    Body *hPos = (Body*)malloc(bytes);
    Body *hVel = (Body*)malloc(bytes);
    initGalaxy(hPos, hVel, N);

    Body *dPos, *dVel, *dAcc;
    CUDA_CHECK(cudaMalloc(&dPos, bytes));
    CUDA_CHECK(cudaMalloc(&dVel, bytes));
    CUDA_CHECK(cudaMalloc(&dAcc, bytes));
    CUDA_CHECK(cudaMemcpy(dPos, hPos, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dVel, hVel, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dAcc, 0, bytes));

    // ── VBO compartido CUDA-OpenGL ─────────────────────────────────────────
    // El VBO vive en la VRAM de la GPU.
    // CUDA escribe las posiciones directamente al VBO cada frame.
    // OpenGL lee el mismo VBO para render → cero cudaMemcpy.
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, N * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Atributo 0: posicion xyz
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // Atributo 1: masa (para tamaño del punto)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(3*sizeof(float)));
    glEnableVertexAttribArray(1);

    // Registrar VBO en CUDA
    cudaGraphicsResource* cudaVBORes;
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaVBORes, vbo,
                                            cudaGraphicsMapFlagsWriteDiscard));

    // ── Shader program ─────────────────────────────────────────────────────
    GLuint prog = buildProgram(VS_SOURCE, FS_SOURCE);
    GLint locMVP     = glGetUniformLocation(prog, "uMVP");
    GLint locMaxMass = glGetUniformLocation(prog, "uMaxMass");
    GLint locMaxDist = glGetUniformLocation(prog, "uMaxDist");

    // ── Estado OpenGL ──────────────────────────────────────────────────────
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);   // blending aditivo: zonas densas brillan
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_POINT_SPRITE);
    glDisable(GL_DEPTH_TEST);            // transparencia aditiva no necesita depth

    // ── Configuracion de kernels ───────────────────────────────────────────
    int gridSim  = (N + TILE_SIZE - 1) / TILE_SIZE;
    int gridCopy = (N + 255) / 256;
    size_t shMem = TILE_SIZE * sizeof(Body);

    // Benchmark inicial
    if (N <= 10000) benchmarkCPUvsGPU(N);

    printf("Iniciando loop de simulacion...\n");
    g_app.lastFPSTime = glfwGetTime();

    // ─────────────────────────────────────────────────────────────────────
    //  LOOP PRINCIPAL
    // ─────────────────────────────────────────────────────────────────────
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        // ── Paso de fisica en GPU ─────────────────────────────────────────
        if (!g_app.paused) {
            // 1. Calcular fuerzas (O(N^2) con tiling)
            kernelFuerzas<<<gridSim, TILE_SIZE, shMem>>>(dPos, dAcc, N);

            // 2. Integrar (Leapfrog)
            kernelLeapfrog<<<gridSim, TILE_SIZE>>>(dPos, dVel, dAcc, N, DT);

            g_app.steps++;
            g_app.simTime += DT;
        }

        // ── Copiar posiciones al VBO (CUDA→OpenGL sin pasar por CPU) ──────
        float* devVBOPtr;
        size_t vboSize;
        CUDA_CHECK(cudaGraphicsMapResources(1, &cudaVBORes, 0));
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(
                       (void**)&devVBOPtr, &vboSize, cudaVBORes));

        kernelCopiaVBO<<<gridCopy, 256>>>(dPos, devVBOPtr, N);

        CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaVBORes, 0));

        // ── Render ────────────────────────────────────────────────────────
        int fbW, fbH;
        glfwGetFramebufferSize(window, &fbW, &fbH);
        glViewport(0, 0, fbW, fbH);
        glClearColor(0.0f, 0.0f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);

        // Matriz MVP: Perspectiva * Traslacion * RotY * RotX
        Mat4 proj, view, rx, ry, mvp, tmp;
        mat4Perspective(proj, 0.85f, (float)fbW/fbH, 0.1f, 200.0f);
        mat4Trans(view, 0.0f, -1.5f, -g_cam.zoom);
        mat4RotX(rx, g_cam.rotX);
        mat4RotY(ry, g_cam.rotY);

        mat4Mul(tmp, ry, rx);
        mat4Mul(mvp, view, tmp);
        mat4Mul(mvp, proj, mvp);

        glUniformMatrix4fv(locMVP, 1, GL_FALSE, mvp);
        glUniform1f(locMaxMass, 0.35f);
        glUniform1f(locMaxDist, 14.0f);

        // Auto-rotacion suave
        if (!g_cam.dragging)
            g_cam.rotY += 0.0003f;

        glBindVertexArray(vao);
        glDrawArrays(GL_POINTS, 0, N);

        // ── FPS en titulo ─────────────────────────────────────────────────
        g_app.fpsFrames++;
        double now = glfwGetTime();
        if (now - g_app.lastFPSTime >= 0.5) {
            g_app.fps = (float)(g_app.fpsFrames / (now - g_app.lastFPSTime));
            g_app.fpsFrames = 0;
            g_app.lastFPSTime = now;
            snprintf(title, sizeof(title),
                     "Galaxy N-body | N=%d | %.0f FPS | paso=%ld | t=%.2f",
                     N, g_app.fps, g_app.steps, g_app.simTime);
            glfwSetWindowTitle(window, title);
        }

        glfwSwapBuffers(window);
    }

    // ── Limpieza ───────────────────────────────────────────────────────────
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaVBORes));
    cudaFree(dPos); cudaFree(dVel); cudaFree(dAcc);
    free(hPos); free(hVel);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glfwDestroyWindow(window);
    glfwTerminate();

    printf("Simulacion terminada. Pasos: %ld, tiempo simulado: %.4f\n",
           g_app.steps, g_app.simTime);
    return 0;
}
