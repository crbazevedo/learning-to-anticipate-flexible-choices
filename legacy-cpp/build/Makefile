# W18-1 Cross-validation harness — C++ legacy build adapter
#
# Builds the C++ ASMOO reference implementation on Apple silicon /
# Apple clang 17 / C++14 / Eigen 3.4. Patches are PURELY mechanical
# build adapters (see README-BUILD-ADAPTER.md); they do NOT change
# algorithm semantics.
#
# Usage:
#   make all          # build all .o + drivers
#   make eigen        # download + extract Eigen 3.4.0 if missing
#   make patches      # apply build adapters to legacy-cpp/source + headers
#   make objs         # compile all source files to objs/
#   make driver-NAME  # build cross-check driver: drivers/NAME_driver
#   make clean        # remove build artifacts (keep eigen download)
#   make distclean    # remove everything except this Makefile

CXX            = g++
CXXFLAGS       = -std=c++14 -DEIGEN_DONT_VECTORIZE -O2 -Wno-deprecated-declarations
EIGEN_VERSION  = 3.4.0
EIGEN_DIR      = eigen-$(EIGEN_VERSION)
EIGEN_URL      = https://gitlab.com/libeigen/eigen/-/archive/$(EIGEN_VERSION)/eigen-$(EIGEN_VERSION).tar.gz
BOOST_INC      = /opt/homebrew/include
INCLUDES       = -I./headers_patched -I$(BOOST_INC) -I./$(EIGEN_DIR)

SOURCES        = aprendizado_operadores kalman_filter mvtnorm nsga2 \
                  operadores_cruzamento operadores_mutacao operadores_selecao \
                  portfolio statistics utils
OBJS           = $(addprefix objs/, $(addsuffix .o, $(SOURCES)))

DRIVER_SRCS    = $(wildcard drivers/*_driver.cpp)
DRIVER_BINS    = $(patsubst drivers/%.cpp, drivers/%, $(DRIVER_SRCS))


.PHONY: all eigen patches objs drivers clean distclean

all: objs drivers

eigen: $(EIGEN_DIR)/Eigen/Dense

$(EIGEN_DIR)/Eigen/Dense:
	@echo "[W18-1] Downloading Eigen $(EIGEN_VERSION)..."
	@if [ ! -f $(EIGEN_DIR).tar.gz ]; then \
		curl -fsSL $(EIGEN_URL) -o $(EIGEN_DIR).tar.gz; \
	fi
	@tar xzf $(EIGEN_DIR).tar.gz
	@echo "[W18-1] Eigen $(EIGEN_VERSION) installed at ./$(EIGEN_DIR)"

# W18-1 ADAPTER 1+2: Eigen include path + double->uint narrowing.
# Creates src_patched/ and headers_patched/ from frozen legacy source.
patches: src_patched/.timestamp headers_patched/.timestamp

src_patched/.timestamp: ../source/*.cpp
	@echo "[W18-1] Patching ../source -> src_patched (Eigen include + narrowing)..."
	@rm -rf src_patched
	@cp -r ../source src_patched
	@find src_patched -name "*.cpp" | xargs sed -i '' \
		-e 's|<Eigen/Eigen/|<Eigen/|g' \
		-e 's|"../headers/|"|g' \
		-e 's|2\.0\*portfolio::window_size|2u*portfolio::window_size|g'
	@touch src_patched/.timestamp

headers_patched/.timestamp: ../headers/*.h
	@echo "[W18-1] Patching ../headers -> headers_patched..."
	@rm -rf headers_patched
	@cp -r ../headers headers_patched
	@find headers_patched -name "*.h" | xargs sed -i '' \
		-e 's|<Eigen/Eigen/|<Eigen/|g' \
		-e 's|"../headers/|"|g' \
		-e 's|2\.0\*portfolio::window_size|2u*portfolio::window_size|g'
	@touch headers_patched/.timestamp

objs: eigen patches $(OBJS)

objs/%.o: src_patched/%.cpp | objs_dir
	@echo "[W18-1] CXX $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

objs_dir:
	@mkdir -p objs

# Per-check driver builds. Each cross-check ships a single
# drivers/NAME_driver.cpp; the rule links it against a CURATED subset
# of legacy objects required for that check. Avoids pulling in mvtnorm.o
# (which depends on Fortran `_mvtdst_` symbol we don't have on Apple
# silicon for now) when the driver doesn't need it.
#
# By default driver-X links against portfolio + utils + statistics +
# kalman_filter (the foundational stack). Drivers needing additional
# modules can override OBJS_FOR_<name>.
DRIVER_BASE_OBJS = $(OBJS) objs/mvtdst_stub.o

# mvtdst_stub: provide a stub for the Fortran `mvtdst_` symbol so we
# can link drivers that don't actually invoke multivariate normal CDF
# (most cross-checks). If a driver DOES call it (e.g. TIP), it will
# get zeros + INFORM=99; the driver must check INFORM and fall back
# to its own MC implementation or skip.
objs/mvtdst_stub.o: drivers/mvtdst_stub.cpp
	@echo "[W18-1] CXX $<"
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# TIP driver (W18-4) needs the multivariate normal stack too — that's the
# mvtnorm-dependent path. We'll handle that with a separate rule when it lands.

drivers: $(DRIVER_BINS)

drivers/%: drivers/%.cpp $(DRIVER_BASE_OBJS)
	@echo "[W18-1] LD $@"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(DRIVER_BASE_OBJS) -o $@

clean:
	@echo "[W18-1] cleaning build artifacts (keeping eigen download)..."
	@rm -rf objs/ src_patched/ headers_patched/
	@rm -f drivers/*_driver
	@rm -f drivers/*.o

distclean: clean
	@rm -rf $(EIGEN_DIR) $(EIGEN_DIR).tar.gz
