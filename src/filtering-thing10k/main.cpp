#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/vertex_position_geometry.h"

#include "geometrycentral/surface/direction_fields.h"

#include "args/args.hxx"
#include "imgui.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

// == Geometry-central data
std::unique_ptr<SurfaceMesh> mesh;
std::unique_ptr<VertexPositionGeometry> geometry;

int main(int argc, char **argv) {
  //std::cout << "Init\n";
  // Configure the argument parser
  args::ArgumentParser parser("geometry-central & Polyscope example project");
  args::Positional<std::string> inputFilename(parser, "mesh", "A mesh file.");

  //// Parse args
  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &h) {
    std::cout << parser;
    return 0;
  } catch (args::ParseError &e) {
    std::cerr << e.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  // Make sure a mesh name was given
  if (!inputFilename) {
    std::cerr << "Please specify a mesh file as argument" << std::endl;
    return EXIT_FAILURE;
  }

  //std::cout << "Parsed\n";

  // Load mesh
  std::tie(mesh, geometry) = readSurfaceMesh(args::get(inputFilename));

  if(
    !(
      mesh->nConnectedComponents() == 1 &&
      mesh->isManifold() &&
      mesh->isOriented() &&
      mesh->isTriangular()
    )
  )
  {
    std::cout << "1";
    return EXIT_FAILURE;
  }

  int edgeLengthsTight = 0;
  geometry->requireEdgeLengths();
  double avg_edge_length = geometry->edgeLengths.raw().sum() / mesh->nEdges();
  double delta = 0.001 * avg_edge_length;

  for (Face f : mesh->faces())
  {
    auto fE = f.adjacentEdges();
    auto e_i = fE.begin();
    Edge e1 = *e_i;
    ++e_i;
    Edge e2 = *e_i;
    ++e_i;
    Edge e3 = *e_i;
    
    /*std::cout << avg_edge_length << " " << delta << " " 
      << e1.getIndex() << " " << geometry->edgeLengths[e1] << " " 
      << e2.getIndex() << " " << geometry->edgeLengths[e2] << " " 
      << e3.getIndex() << " " << geometry->edgeLengths[e3] << "\n";*/

    if (
      (geometry->edgeLengths[e1] + geometry->edgeLengths[e2] < geometry->edgeLengths[e3] + delta) ||
      (geometry->edgeLengths[e1] + geometry->edgeLengths[e3] < geometry->edgeLengths[e2] + delta) ||
      (geometry->edgeLengths[e2] + geometry->edgeLengths[e3] < geometry->edgeLengths[e1] + delta)
    )
    {
      edgeLengthsTight++;
      /*
      edgeLengthsTight = true;
      break;*/
    }

  }

  if (edgeLengthsTight > 0.05 * mesh->nFaces()) // > 5%
  {
    std::cout << "4";
    return EXIT_SUCCESS;
  }
  else
  {
    std::cout << "5";
    return EXIT_FAILURE;
  }
}
