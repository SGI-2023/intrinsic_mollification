#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Heat_method_3/Surface_mesh_geodesic_distances_3.h>
#include <CGAL/Surface_mesh_shortest_path.h>
#include <CGAL/Random.h>
#include <boost/lexical_cast.hpp>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>


typedef CGAL::Simple_cartesian<double> Kernel;
typedef Kernel::Point_3 Point_3;
typedef CGAL::Surface_mesh<Point_3> Triangle_mesh;

typedef boost::graph_traits<Triangle_mesh>::vertex_descriptor vertex_descriptor;
typedef Triangle_mesh::Property_map<vertex_descriptor, double> Vertex_distance_map;
typedef CGAL::Heat_method_3::Surface_mesh_geodesic_distances_3<Triangle_mesh, CGAL::Heat_method_3::Intrinsic_Delaunay> Heat_method;

typedef CGAL::Surface_mesh_shortest_path_traits<Kernel, Triangle_mesh> Traits;
typedef CGAL::Surface_mesh_shortest_path<Traits> Surface_mesh_shortest_path;
typedef boost::graph_traits<Triangle_mesh> Graph_traits;
typedef Graph_traits::vertex_iterator vertex_iterator;
typedef Graph_traits::face_iterator face_iterator;

double median(std::vector<double>& v)
{
  if (v.size() % 2)
  {
    return v[v.size() / 2];
  }
  else
  {
    return (v[v.size() / 2 - 1] + v[v.size() / 2]) / 2;
  }
}

int main(int argc, char* argv[])
{
  // we need to benchmark the heat method with different mollifiers and different epsilon values
  std::vector<CGAL::Heat_method_3::MOLLIFICATION_TYPE> mollifiers = {
      CGAL::Heat_method_3::NONE,
      CGAL::Heat_method_3::CONSTANT_EPSILON,
      CGAL::Heat_method_3::LOCAL_CONSTANT_EPSILON,
      CGAL::Heat_method_3::LOCAL_ONE_BY_ONE_STEP,
      CGAL::Heat_method_3::LOCAL_ONE_BY_ONE_INTERPOLATION,
  };

  std::vector<std::string> mollifier_names = {
      "NONE",
      "CONSTANT_EPSILON",
      "LOCAL_CONSTANT_EPSILON",
      "LOCAL_ONE_BY_ONE_STEP",
      "LOCAL_ONE_BY_ONE_INTERPOLATION",
  };

  std::vector<double> epsilons = { 1e-6, 1e-4, 1e-2 };

  // vector of MREs for each mollifier and epsilon and each mesh
  std::vector<std::vector<std::vector<double>>> MREs(mollifiers.size(), std::vector<std::vector<double>>(epsilons.size(), std::vector<double>()));
  std::vector<std::vector<std::vector<double>>> Outliers(mollifiers.size(), std::vector<std::vector<double>>(epsilons.size(), std::vector<double>()));

  // read a txt file "../../../../../../../../../../../4B/GC/tooNiceMeshPaths.txt" containing the paths to the meshes
  // and then read the meshes one by one
  std::string paths_file = "../../../../../../../../../../../4B/GC/tooNiceMeshPaths.txt";

  // read the paths
  std::ifstream paths(paths_file);
  std::string line;
  std::vector<std::string> filenames;

  while (std::getline(paths, line))
  {
    filenames.push_back(line);
  }

  paths_file = "../../../../../../../../../../../4B/GC/validMeshPaths.txt";
  std::ifstream paths2(paths_file);

  // read the paths
  while (std::getline(paths2, line))
  {
    filenames.push_back(line);
  }

  auto rng = std::default_random_engine{};

  // shuffle the filenames using std::shuffle
  std::shuffle(filenames.begin(), filenames.end(), rng);


  std::cout << " number of meshes: " << filenames.size() << "\n";

  // read the meshes
  for (int z = 1; z <= filenames.size(); z++)
  {
    std::string filename = filenames[z - 1];
    // remove whitespace and newlines
    filename.erase(std::remove(filename.begin(), filename.end(), ' '), filename.end());

    const std::string new_filename = "../../../../../../../../../../../4B/GC/Thingi10K/Thingi10K/raw_meshes/" + filename;
    // const std::string filename = (argc > 1) ? argv[1] : "../../../../../../../../../../../4B/GC/Thingi10K/Thingi10K/raw_meshes/104421.stl";
    std::cout << "##############################################\n";
    std::cout << z << " : " << filename << "\n";
    Triangle_mesh tm;

    try {
      if (!CGAL::IO::read_polygon_mesh(new_filename, tm) ||
        CGAL::is_empty(tm) || !CGAL::is_triangle_mesh(tm))
      {
        std::cerr << "Invalid input file." << std::endl;
        return EXIT_FAILURE;
      }
    }
    catch (std::exception e) {
      std::cout << "exception caught: " << e.what() << std::endl;
      continue;
    }

    if (num_faces(tm) > 20000) {
      std::cout << "skipping mesh with " << num_faces(tm) << " faces" << std::endl;
      continue;
    }

    try {

      // Pick a vertex
      int i = 151403424 % num_vertices(tm);
      // add the first vertex at the ith index as source
      vertex_descriptor source = *(vertices(tm).first + i);

      // property map for the distance values to the source set
      Vertex_distance_map heat_vertex_distance = tm.add_property_map<vertex_descriptor, double>("v:distance", 0).first;
      Vertex_distance_map true_vertex_distance = tm.add_property_map<vertex_descriptor, double>("v:true_distance", 0).first;

      // construct a shortest path query object and add a source point
      Surface_mesh_shortest_path shortest_paths(tm);
      shortest_paths.add_source_point(source);
      // For all vertices in the tmesh, compute the points of
      // the shortest path to the source point and write them
      // into a file readable using the CGAL Polyhedron demo
      vertex_iterator vit, vit_end;

      for (boost::tie(vit, vit_end) = vertices(tm);
        vit != vit_end; ++vit)
      {
        double true_distance = shortest_paths.shortest_distance_to_source_points(*vit).first;
        true_vertex_distance[*vit] = true_distance;
      }

      for (auto mollifier : mollifiers)
      {
        std::cout << "______________________________________________\n";
        //std::cout << "epsilon: " << epsilon << std::endl;
        for (int e = 0; e < epsilons.size(); e++)
        {
          double epsilon = epsilons[e];
          //std::cout << "mollifier: " << mollifier_names[mollifier] << std::endl;
          Heat_method hm(tm, mollifier, epsilon);

          hm.add_source(source);
          hm.estimate_geodesic_distances(heat_vertex_distance);

          double MRE = 0;
          bool inf_nan = false;

          for (boost::tie(vit, vit_end) = vertices(tm);
            vit != vit_end; ++vit)
          {
            double true_distance = true_vertex_distance[*vit];
            double heat_distance = heat_vertex_distance[*vit];

            // compute Relative error
            double re;
            if (true_distance < 1e-10)
            {
              re = std::abs(true_distance - heat_distance) / 1e-10;
            }
            else
            {
              re = std::abs(true_distance - heat_distance) / true_distance;
            }
            MRE += re;
            if (isinf(re) || isnan(re))
            {
              // std::cout << re << " ";
              inf_nan = true;
              break;
            }
          }

          if (inf_nan)
          {
            Outliers[mollifier][e].push_back(MRE / num_vertices(tm));
          }
          else if (MRE / num_vertices(tm) > 1)
          {
            Outliers[mollifier][e].push_back(MRE / num_vertices(tm));
          }
          else
          {
            MREs[mollifier][e].push_back(MRE / num_vertices(tm));
          }
          //std::cout << "MRE: " << MRE / n << " " << n << std::endl;
        }
      }

    }
    catch (std::exception e) {
      std::cout << "exception caught: " << e.what() << std::endl;
      continue;
    }

    try {


      if (z % 10 == 0) {
        std::cout << "______________________________________________\n";
        std::cout << "completed " << z << " meshes" << std::endl;
        for (auto mollifier : mollifiers)
        {
          std::cout << "\nmollifier: " << mollifier_names[mollifier] << std::endl;
          for (int e = 0; e < epsilons.size(); e++)
          {
            double epsilon = epsilons[e];
            std::cout << "\nepsilon: " << epsilon << std::endl;
            int inf_count = 0;
            for (double outlier : Outliers[mollifier][e])
            {
              if (isinf(outlier) || isnan(outlier))
              {
                inf_count++;
              }
            }

            // make a vector containing MREs and Outliers (concatenate the two vectors),
            std::vector<double> all_MREs(MREs[mollifier][e].size() + Outliers[mollifier][e].size());
            std::copy(MREs[mollifier][e].begin(), MREs[mollifier][e].end(), all_MREs.begin());
            std::copy(Outliers[mollifier][e].begin(), Outliers[mollifier][e].end(), all_MREs.begin() + MREs[mollifier][e].size());
            std::sort(all_MREs.begin(), all_MREs.end());
            std::cout << "inf|nan count: " << inf_count << " / " << all_MREs.size() << std::endl;
            std::cout << "outlier (+inf|nan) count: " << Outliers[mollifier][e].size() << " / " << all_MREs.size() << std::endl;
            std::cout << "Median MRE: " << median(all_MREs) << std::endl;
            std::cout << "Mean MRE: " << std::accumulate(MREs[mollifier][e].begin(), MREs[mollifier][e].end(), 0.0) / MREs[mollifier][e].size() << std::endl;
          }
        }
      }
    }

    catch (std::exception e) {
      std::cout << "exception caught: " << e.what() << std::endl;
      continue;
    }
  }


}
