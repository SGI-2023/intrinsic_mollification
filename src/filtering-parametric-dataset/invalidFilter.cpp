#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "geometrycentral/surface/direction_fields.h"

#include <string>
#include <iostream>

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
std::unique_ptr<SurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

bool property(const std::string& option)
{
	if(option == "connected")
		return (mesh->nConnectedComponents() == 1);
	else if(option == "manifold")
		return (mesh->isManifold());
	else if(option == "oriented")
		return (mesh->isOriented());
	else if(option == "triangular")
		return (mesh->isTriangular());
	else
		std::cerr << "Invalid option." << std::endl;

	return false;
}

int main(int argc, char **argv) 
{
	if(argc != 3){
		std::cerr << "Usage is: " << argv[0] << " <option> <path to file>" << std::endl;
		std::cerr << "Option: connected (default), manifold, oriented, triangular."
							<< std::endl;
		return EXIT_FAILURE;
	}

	std::string option = argv[1];
	std::string path = argv[2];

	std::tie(mesh, geometry) = readSurfaceMesh(path);

	if(!(property(option)))
		std::cout << "1";
	else 
		std::cout << "0";

	return EXIT_SUCCESS;
}
