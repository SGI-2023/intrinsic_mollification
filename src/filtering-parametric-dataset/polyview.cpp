#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh.h"

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <string>

using namespace geometrycentral;
using namespace geometrycentral::surface;

int main(int argc, char* argv[])
{
	if(argc != 2){
		std::cout << "Usage: " << argv[0] << " <path to file>" << std::endl;
		return EXIT_FAILURE;
	}
	std::string filename = argv[1];

	polyscope::init();

	std::unique_ptr<SurfaceMesh> mesh;
	std::unique_ptr<VertexPositionGeometry> geometry;
	std::tie(mesh, geometry) = readSurfaceMesh(filename);
	geometry->requireVertexPositions();

	auto psMesh = polyscope::registerSurfaceMesh("input mesh",
																							 geometry->vertexPositions,
																							 mesh->getFaceVertexList());
	polyscope::show();

	return 0;
}
