#include <cstdlib>
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <glad/glad.h>

#include <GLFW/glfw3.h>

#include "gameOfLife/cpu.hpp"
#include "gameOfLife/cuda.hpp"
#include "gameOfLife/interface.hpp"
#include "gameOfLife/opencl.hpp"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

void framebuffer_size_callback(GLFWwindow *window, int width, int height);
void processInput(GLFWwindow *window);
std::vector<float> vertices;
unsigned int VBO, VAO, EBO;
std::map<int, bool> key_press;
std::unique_ptr<GameOfLifeInterface> gol;
void tick_vertices();
void random_grid();
void set_gol(std::vector<std::vector<int>> &grid);
void set_vertices();

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 800;

const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aColor;
out vec3 ourColor;
void main() {
    gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);
    ourColor = aColor;
}
)";
const char *fragmentShaderSource = R"(
#version 330 core
in vec3 ourColor;
out vec4 FragColor;
void main() {
    FragColor = vec4(ourColor, 1.0);
}
)";

int N = 100;
int M = 100;
std::vector<std::vector<int>> grid(N, std::vector<int>(M, 1));

// bool args
bool cuda = false;
bool opencl = false;
bool cpu = true;
int workgroup_x = 16;
int workgroup_y = 16;
int main(int argc, char **argv) {
  for (int i = 0; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--workgroup-x") {
      std::string val = argv[i+1];
      workgroup_x = std::stoi(val);
    }
    if (arg == "--workgroup-y") {
      std::string val = argv[i+1];
      workgroup_y = std::stoi(val);
    }
  }
  // glfw: initialize and configure
  // ------------------------------
  glfwInit();
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // glfw window creation
  // --------------------
  GLFWwindow *window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "Game Of Life", NULL, NULL);
  if (window == NULL) {
    std::cout << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad: load all OpenGL function pointers
  // ---------------------------------------
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cout << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // ImGui
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO &io = ImGui::GetIO();
  (void)io;
  ImGui::StyleColorsDark();
  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  // build and compile our shader program
  // ------------------------------------
  // vertex shader
  unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
  glCompileShader(vertexShader);
  // check for shader compile errors
  int success;
  char infoLog[512];
  glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
  }
  // fragment shader
  unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
  glCompileShader(fragmentShader);
  // check for shader compile errors
  glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
  }
  // link shaders
  unsigned int shaderProgram = glCreateProgram();
  glAttachShader(shaderProgram, vertexShader);
  glAttachShader(shaderProgram, fragmentShader);
  glLinkProgram(shaderProgram);
  // check for linking errors
  glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
    std::cout << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
  }
  glDeleteShader(vertexShader);
  glDeleteShader(fragmentShader);

  // set up vertex data (and buffer(s)) and configure vertex attributes
  // ------------------------------------------------------------------
  random_grid();
  set_gol(grid);
  set_vertices();
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);
  // bind the Vertex Array Object first, then bind and set vertex buffer(s), and
  // then configure vertex attributes(s).
  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  // glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
  //              GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (void *)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  // note that this is allowed, the call to glVertexAttribPointer registered VBO
  // as the vertex attribute's bound vertex buffer object so afterwards we can
  // safely unbind
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // remember: do NOT unbind the EBO while a VAO is active as the bound element
  // buffer object IS stored in the VAO; keep the EBO bound.
  // glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  // You can unbind the VAO afterwards so other VAO calls won't accidentally
  // modify this VAO, but this rarely happens. Modifying other VAOs requires a
  // call to glBindVertexArray anyways so we generally don't unbind VAOs (nor
  // VBOs) when it's not directly necessary.
  glBindVertexArray(0);

  // uncomment this call to draw in wireframe polygons.
  // glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  // render loop
  // -----------
  glfwSwapInterval(0); // Disable VSync (uncaps the frame rate)
  while (!glfwWindowShouldClose(window)) {

    // input
    processInput(window);

    // render
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // draw our first triangle
    glUseProgram(shaderProgram);
    glBindVertexArray(VAO); // seeing as we only have a single VAO there's no need to bind it
                            // every time, but we'll do so to keep things a bit more organized
    glDrawArrays(GL_TRIANGLES, 0, vertices.size() / 6);
    // ImGui
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    auto curr_grid = gol->get_grid();
    // Build UI
    ImGui::Begin("Game Of Life Controls");
    ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
    ImGui::Text("Cells/s: %.0f", ImGui::GetIO().Framerate * N * M);
    if (ImGui::SliderInt("Rows", &M, 1, 1000) | ImGui::SliderInt("Columns", &N, 1, 1000)) {
      random_grid();
      set_gol(grid);
      set_vertices();
    }
    if (ImGui::Button("CPU")) {
      cpu = true;
      cuda = false;
      opencl = false;
      set_gol(curr_grid);
    }
    ImGui::SameLine();
    if (ImGui::Button("OpenCL")) {
      cpu = false;
      cuda = false;
      opencl = true;
      set_gol(curr_grid);
    }
    ImGui::SameLine();
    if (ImGui::Button("Cuda")) {
      cpu = false;
      cuda = true;
      opencl = false;
      set_gol(curr_grid);
    }
    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    tick_vertices();

    // glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    // glBindVertexArray(0); // no need to unbind it every time

    // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved
    // etc.)
    // -------------------------------------------------------------------------------
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();

  // optional: de-allocate all resources once they've outlived their purpose:
  // ------------------------------------------------------------------------
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
  glDeleteProgram(shaderProgram);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();
  return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this
// frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS && !key_press[GLFW_KEY_SPACE]) {
    key_press[GLFW_KEY_SPACE] = true;
    cpu = !cpu;
    auto curr_grid = gol->get_grid();
    if (cpu) {
      gol = std::make_unique<GameOfLifeCPU>(curr_grid);
    } else {
      gol = std::make_unique<GameOfLifeOpenCL>(curr_grid, workgroup_x, workgroup_y);
    }
  }
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_RELEASE) {
    key_press[GLFW_KEY_SPACE] = false;
  }
}

// glfw: whenever the window size changed (by OS or user resize) this callback
// function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow *window, int width, int height) {
  // make sure the viewport matches the new window dimensions; note that width
  // and height will be significantly larger than specified on retina displays.
  int new_width = std::min(width, height);
  int new_height = new_width;
  int x = 0;
  int y = 0;
  if (width > height) {
    x = (width - height) / 2;
  }
  if (height > width) {
    y = (height - width) / 2;
  }
  glViewport(x, y, new_width, new_height);
}
void tick_vertices() {
  gol->tick();
  auto curr_grid = gol->get_grid();
  for (int i = 0; i < curr_grid.size(); i++) {
    for (int j = 0; j < curr_grid[0].size(); j++) {
      float v = curr_grid[i][j] ? 1.0f : 0.3f;
      int base_index = (i * curr_grid[0].size() + j) * 6 * 6;
      for (int vertex = base_index; vertex < base_index + 6 * 6; vertex += 6) {
        vertices[vertex + 3] = v;
        vertices[vertex + 4] = v;
        vertices[vertex + 5] = v;
      }
    }
  }
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(float), vertices.data());
}
void random_grid() {
  grid.resize(N);
  for (auto &v : grid) {
    v.resize(M);
  }
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      grid[i][j] = rand() % 2 == 0;
    }
  }
}
void set_gol(std::vector<std::vector<int>> &grid) {
  if (cpu)
    gol = std::make_unique<GameOfLifeCPU>(grid);
  if (cuda)
    gol = std::make_unique<GameOfLifeCuda>(grid);
  if (opencl)
    gol = std::make_unique<GameOfLifeOpenCL>(grid, workgroup_x, workgroup_y);
}
void set_vertices() {
  vertices.clear();
  float gap_frac = 0.20f;
  float step = 2.0f / (std::max(N, M) + 1);

  float grid_width = step * N;
  float grid_height = step * M;

  float x_offset = -grid_width / 2.0f;
  float y_offset = -grid_height / 2.0f;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < M; j++) {
      float pos_x = x_offset + step * (i + 0.5f);
      float pos_y = y_offset + step * (j + 0.5f);

      float gap = step / 2 - step * gap_frac;
      float left = pos_x - gap;
      float right = pos_x + gap;
      float top = pos_y + gap;
      float bottom = pos_y - gap;

      float r = grid[i][j] ? 1.0f : 0.3f;
      float g = r;
      float b = r;
      // clang-format off
      vertices.insert(vertices.end(), {
          right, top, 0.0f, r, g, b,
          right, bottom, 0.0f, r, g, b,
          left, bottom, 0.0f, r, g, b,
      });
      vertices.insert(vertices.end(), {
          left, top, 0.0f, r, g, b,
          left, bottom, 0.0f, r, g, b,
          right, top, 0.0f, r, g, b,
      });
      // clang-format on
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_DYNAMIC_DRAW);
}
