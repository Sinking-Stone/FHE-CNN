# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.26

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/wangfulei/.local/lib/python3.11/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/wangfulei/.local/lib/python3.11/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/wangfulei/FHE-MP-CNN/cnn_ckks

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/wangfulei/FHE-MP-CNN/cnn_ckks/build

# Include any dependencies generated for this target.
include CMakeFiles/cnn.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/cnn.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/cnn.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cnn.dir/flags.make

CMakeFiles/cnn.dir/run/run_cnn.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/run/run_cnn.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/run/run_cnn.cpp
CMakeFiles/cnn.dir/run/run_cnn.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cnn.dir/run/run_cnn.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/run/run_cnn.cpp.o -MF CMakeFiles/cnn.dir/run/run_cnn.cpp.o.d -o CMakeFiles/cnn.dir/run/run_cnn.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/run/run_cnn.cpp

CMakeFiles/cnn.dir/run/run_cnn.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/run/run_cnn.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/run/run_cnn.cpp > CMakeFiles/cnn.dir/run/run_cnn.cpp.i

CMakeFiles/cnn.dir/run/run_cnn.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/run/run_cnn.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/run/run_cnn.cpp -o CMakeFiles/cnn.dir/run/run_cnn.cpp.s

CMakeFiles/cnn.dir/common/Choosemax.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/Choosemax.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Choosemax.cpp
CMakeFiles/cnn.dir/common/Choosemax.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cnn.dir/common/Choosemax.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/Choosemax.cpp.o -MF CMakeFiles/cnn.dir/common/Choosemax.cpp.o.d -o CMakeFiles/cnn.dir/common/Choosemax.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Choosemax.cpp

CMakeFiles/cnn.dir/common/Choosemax.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/Choosemax.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Choosemax.cpp > CMakeFiles/cnn.dir/common/Choosemax.cpp.i

CMakeFiles/cnn.dir/common/Choosemax.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/Choosemax.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Choosemax.cpp -o CMakeFiles/cnn.dir/common/Choosemax.cpp.s

CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompFunc.cpp
CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o -MF CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o.d -o CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompFunc.cpp

CMakeFiles/cnn.dir/common/MinicompFunc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/MinicompFunc.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompFunc.cpp > CMakeFiles/cnn.dir/common/MinicompFunc.cpp.i

CMakeFiles/cnn.dir/common/MinicompFunc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/MinicompFunc.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompFunc.cpp -o CMakeFiles/cnn.dir/common/MinicompFunc.cpp.s

CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompRemez.cpp
CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o -MF CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o.d -o CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompRemez.cpp

CMakeFiles/cnn.dir/common/MinicompRemez.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/MinicompRemez.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompRemez.cpp > CMakeFiles/cnn.dir/common/MinicompRemez.cpp.i

CMakeFiles/cnn.dir/common/MinicompRemez.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/MinicompRemez.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/MinicompRemez.cpp -o CMakeFiles/cnn.dir/common/MinicompRemez.cpp.s

CMakeFiles/cnn.dir/common/Point.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/Point.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Point.cpp
CMakeFiles/cnn.dir/common/Point.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/cnn.dir/common/Point.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/Point.cpp.o -MF CMakeFiles/cnn.dir/common/Point.cpp.o.d -o CMakeFiles/cnn.dir/common/Point.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Point.cpp

CMakeFiles/cnn.dir/common/Point.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/Point.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Point.cpp > CMakeFiles/cnn.dir/common/Point.cpp.i

CMakeFiles/cnn.dir/common/Point.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/Point.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Point.cpp -o CMakeFiles/cnn.dir/common/Point.cpp.s

CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/PolyUpdate.cpp
CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o -MF CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o.d -o CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/PolyUpdate.cpp

CMakeFiles/cnn.dir/common/PolyUpdate.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/PolyUpdate.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/PolyUpdate.cpp > CMakeFiles/cnn.dir/common/PolyUpdate.cpp.i

CMakeFiles/cnn.dir/common/PolyUpdate.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/PolyUpdate.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/PolyUpdate.cpp -o CMakeFiles/cnn.dir/common/PolyUpdate.cpp.s

CMakeFiles/cnn.dir/common/func.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/func.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/func.cpp
CMakeFiles/cnn.dir/common/func.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/cnn.dir/common/func.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/func.cpp.o -MF CMakeFiles/cnn.dir/common/func.cpp.o.d -o CMakeFiles/cnn.dir/common/func.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/func.cpp

CMakeFiles/cnn.dir/common/func.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/func.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/func.cpp > CMakeFiles/cnn.dir/common/func.cpp.i

CMakeFiles/cnn.dir/common/func.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/func.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/func.cpp -o CMakeFiles/cnn.dir/common/func.cpp.s

CMakeFiles/cnn.dir/common/Polynomial.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/Polynomial.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Polynomial.cpp
CMakeFiles/cnn.dir/common/Polynomial.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/cnn.dir/common/Polynomial.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/Polynomial.cpp.o -MF CMakeFiles/cnn.dir/common/Polynomial.cpp.o.d -o CMakeFiles/cnn.dir/common/Polynomial.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Polynomial.cpp

CMakeFiles/cnn.dir/common/Polynomial.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/Polynomial.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Polynomial.cpp > CMakeFiles/cnn.dir/common/Polynomial.cpp.i

CMakeFiles/cnn.dir/common/Polynomial.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/Polynomial.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Polynomial.cpp -o CMakeFiles/cnn.dir/common/Polynomial.cpp.s

CMakeFiles/cnn.dir/common/Remez.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/Remez.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Remez.cpp
CMakeFiles/cnn.dir/common/Remez.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/cnn.dir/common/Remez.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/Remez.cpp.o -MF CMakeFiles/cnn.dir/common/Remez.cpp.o.d -o CMakeFiles/cnn.dir/common/Remez.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Remez.cpp

CMakeFiles/cnn.dir/common/Remez.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/Remez.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Remez.cpp > CMakeFiles/cnn.dir/common/Remez.cpp.i

CMakeFiles/cnn.dir/common/Remez.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/Remez.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/Remez.cpp -o CMakeFiles/cnn.dir/common/Remez.cpp.s

CMakeFiles/cnn.dir/common/RemezApp.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/common/RemezApp.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/RemezApp.cpp
CMakeFiles/cnn.dir/common/RemezApp.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/cnn.dir/common/RemezApp.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/common/RemezApp.cpp.o -MF CMakeFiles/cnn.dir/common/RemezApp.cpp.o.d -o CMakeFiles/cnn.dir/common/RemezApp.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/RemezApp.cpp

CMakeFiles/cnn.dir/common/RemezApp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/common/RemezApp.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/RemezApp.cpp > CMakeFiles/cnn.dir/common/RemezApp.cpp.i

CMakeFiles/cnn.dir/common/RemezApp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/common/RemezApp.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/common/RemezApp.cpp -o CMakeFiles/cnn.dir/common/RemezApp.cpp.s

CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/cnn_seal.cpp
CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o -MF CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o.d -o CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/cnn_seal.cpp

CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/cnn_seal.cpp > CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.i

CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/cnn_seal.cpp -o CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.s

CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp
CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o -MF CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o.d -o CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp

CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp > CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.i

CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/cnn/infer_seal.cpp -o CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.s

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/program.cpp
CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o -MF CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o.d -o CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/program.cpp

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/program.cpp > CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.i

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/program.cpp -o CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.s

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALcomp.cpp
CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o -MF CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o.d -o CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALcomp.cpp

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALcomp.cpp > CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.i

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALcomp.cpp -o CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.s

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALfunc.cpp
CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o -MF CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o.d -o CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALfunc.cpp

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALfunc.cpp > CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.i

CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/comp/SEALfunc.cpp -o CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.s

CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp
CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o -MF CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o.d -o CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp

CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp > CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.i

CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp -o CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.s

CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o: CMakeFiles/cnn.dir/flags.make
CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o: /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp
CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o: CMakeFiles/cnn.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o -MF CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o.d -o CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o -c /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp

CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp > CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.i

CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/wangfulei/FHE-MP-CNN/cnn_ckks/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp -o CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.s

# Object files for target cnn
cnn_OBJECTS = \
"CMakeFiles/cnn.dir/run/run_cnn.cpp.o" \
"CMakeFiles/cnn.dir/common/Choosemax.cpp.o" \
"CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o" \
"CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o" \
"CMakeFiles/cnn.dir/common/Point.cpp.o" \
"CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o" \
"CMakeFiles/cnn.dir/common/func.cpp.o" \
"CMakeFiles/cnn.dir/common/Polynomial.cpp.o" \
"CMakeFiles/cnn.dir/common/Remez.cpp.o" \
"CMakeFiles/cnn.dir/common/RemezApp.cpp.o" \
"CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o" \
"CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o" \
"CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o" \
"CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o" \
"CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o" \
"CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o" \
"CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o"

# External object files for target cnn
cnn_EXTERNAL_OBJECTS =

run/cnn: CMakeFiles/cnn.dir/run/run_cnn.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/Choosemax.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/MinicompFunc.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/MinicompRemez.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/Point.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/PolyUpdate.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/func.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/Polynomial.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/Remez.cpp.o
run/cnn: CMakeFiles/cnn.dir/common/RemezApp.cpp.o
run/cnn: CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/cnn_seal.cpp.o
run/cnn: CMakeFiles/cnn.dir/cpu-ckks/single-key/cnn/infer_seal.cpp.o
run/cnn: CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/program.cpp.o
run/cnn: CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALcomp.cpp.o
run/cnn: CMakeFiles/cnn.dir/cpu-ckks/single-key/comp/SEALfunc.cpp.o
run/cnn: CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/Bootstrapper.cpp.o
run/cnn: CMakeFiles/cnn.dir/cpu-ckks/single-key/ckks_bootstrapping/ModularReducer.cpp.o
run/cnn: CMakeFiles/cnn.dir/build.make
run/cnn: /usr/local/lib/libseal-3.6.a
run/cnn: CMakeFiles/cnn.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Linking CXX executable run/cnn"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnn.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cnn.dir/build: run/cnn
.PHONY : CMakeFiles/cnn.dir/build

CMakeFiles/cnn.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cnn.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cnn.dir/clean

CMakeFiles/cnn.dir/depend:
	cd /home/wangfulei/FHE-MP-CNN/cnn_ckks/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/wangfulei/FHE-MP-CNN/cnn_ckks /home/wangfulei/FHE-MP-CNN/cnn_ckks /home/wangfulei/FHE-MP-CNN/cnn_ckks/build /home/wangfulei/FHE-MP-CNN/cnn_ckks/build /home/wangfulei/FHE-MP-CNN/cnn_ckks/build/CMakeFiles/cnn.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cnn.dir/depend
