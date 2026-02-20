cmake_minimum_required(VERSION 3.20)
project(proxy_server VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# Boost (Beast is header-only; we need system + thread for Asio)
find_package(Boost 1.74 REQUIRED COMPONENTS system thread)

# nlohmann/json (header-only JSON library)
find_package(nlohmann_json 3.10 QUIET)
if(NOT nlohmann_json_FOUND)
    # Fallback: use the system package installed via apt
    include_directories(/usr/include/nlohmann)
endif()

# Sources
set(SOURCES
    src/main.cpp
    src/server.cpp
    src/queue.cpp
    src/metrics.cpp
    src/uds_client.cpp
)

add_executable(proxy_server ${SOURCES})

target_include_directories(proxy_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${Boost_INCLUDE_DIRS}
)

target_link_libraries(proxy_server PRIVATE
    Boost::system
    Boost::thread
    pthread
)

# Installation
install(TARGETS proxy_server DESTINATION bin)
