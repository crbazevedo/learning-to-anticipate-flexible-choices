# legacy-cpp-v2/build/Makefile — 2015 ASMOO C++ adapter (W18-substrate-update follow-up).
#
# Builds the v2 paper-companion C++ reference on Apple silicon /
# clang 17 / C++14 / Eigen 3.4. Adapters are PURELY mechanical
# (no algorithm changes); see ../README.md for the diff vs v1.
#
# Usage:
#   make patches  # apply build adapters (sed includes + casts) to a fresh copy
#   make objs     # compile core source files
#   make drivers  # build any drivers/*_driver.cpp
#   make clean

CXX            = g++
CXXFLAGS       = -std=c++14 -DEIGEN_DONT_VECTORIZE -O2 -Wno-deprecated-declarations
# Reuse the Eigen 3.4 download from legacy-cpp/build/ so the substrate-import
# wave doesn't have to download it twice.
EIGEN_DIR      = ../../legacy-cpp/build/eigen-3.4.0
BOOST_INC      = /opt/homebrew/include
INCLUDES       = -I./headers_patched -I$(BOOST_INC) -I$(EIGEN_DIR)

# v2 has more modules than v1 (asms_emoa replaces nsga2; dirichlet new).
# Core modules required by every driver:
CORE_SOURCES   = portfolio kalman_filter statistics utils dirichlet
# Optional modules (drivers that need them link explicitly):
OPTIONAL_SOURCES = asms_emoa crossover_operators mutation_operators \
                    selection_operators learning_operators

CORE_OBJS      = $(addprefix objs/, $(addsuffix .o, $(CORE_SOURCES)))

DRIVER_SRCS    = $(wildcard drivers/*_driver.cpp)
DRIVER_BINS    = $(patsubst drivers/%.cpp, drivers/%, $(DRIVER_SRCS))


.PHONY: all patches objs drivers clean

all: objs drivers

patches: src_patched/.timestamp headers_patched/.timestamp

src_patched/.timestamp: ../source/*.cpp
	@echo "[v2] Patching ../source -> src_patched..."
	@rm -rf src_patched
	@cp -r ../source src_patched
	@find src_patched -name "*.cpp" | xargs sed -i '' \
		-e 's|<Eigen/Eigen/|<Eigen/|g' \
		-e 's|"../headers/|"|g'
	@touch src_patched/.timestamp

headers_patched/.timestamp: ../headers/*.h
	@echo "[v2] Patching ../headers -> headers_patched..."
	@rm -rf headers_patched
	@cp -r ../headers headers_patched
	@find headers_patched -name "*.h" | xargs sed -i '' \
		-e 's|<Eigen/Eigen/|<Eigen/|g' \
		-e 's|"../headers/|"|g'
	@touch headers_patched/.timestamp

objs: patches $(CORE_OBJS)

objs/%.o: src_patched/%.cpp | objs_dir
	@echo "[v2] CXX $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

objs_dir:
	@mkdir -p objs

drivers: objs $(DRIVER_BINS)

drivers/%: drivers/%.cpp $(CORE_OBJS)
	@echo "[v2] LD $@"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(CORE_OBJS) -o $@

clean:
	@echo "[v2] cleaning..."
	@rm -rf objs/ src_patched/ headers_patched/
	@rm -f drivers/*_driver
