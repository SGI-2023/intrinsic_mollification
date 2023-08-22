Vector Distortion::computeQuasiConformalError(const Model& model)
{
	double totalQc = 0.0; // keep track of total distortion
	double minQc = std::numeric_limits<double>::infinity();
	double maxQc = -std::numeric_limits<double>::infinity();
	double totalArea = 1e-8;


        const std::vector<Face>& faces = model[i].faces; // get the list of triangles

        // allocate storage to store one value per face
        distortion.emplace_back(FaceData<double>(model[i])); 
	distortion.clear();
	distortion.reserve(model.size());

        std::vector<Vector> p(3), q(3); // p and q hold local copies of the 3D and 2D vertex coordinates for a single triangle

        for (FaceCIter f = faces.begin(); f != faces.end(); f++) { // iterate over all triangles

           // for the current triangle, grab the three 3D coordinates (p), and the three 2D coordinates (q)
           int j = 0;
           HalfEdgeCIter he = f->halfEdge();
           do {
              p[j] = he->vertex()->position;
              q[j] = he->next()->corner()->uv;
              j++;

              he = he->next();
           } while(he != f->halfEdge());

           // measure the quasi-conformal distortion for this triangle
           double qc = bff::computeQuasiConformalError(p, q);

           double a = area(f); // get the area of this triangle

           // keep track of an area-weighted average of distortion over the whole mesh
           totalQc += qc*a;
           totalArea += a;

           // also keep track of the maximum / minimum distortion in any triangle
           maxQc = std::max(maxQc, qc);
           minQc = std::min(minQc, qc);

           // for visualization purposes, clamp distortion to range [1, 1.5]
           distortion[i][f] = std::max(1.0, std::min(1.5, qc));

        }

        // return the min, max, and mean distortion
	return Vector(minQc, maxQc, totalQc/totalArea);
}

// For a single triangle, takes three 3D coordinates p and three UV coordinates q, and
// returns the quasi-conformal error, i.e., the ratio of max over min stretch.
double computeQuasiConformalError(std::vector<Vector>& p, std::vector<Vector>& q)
{
	// compute edge vectors in 3D
	Vector u1 = p[1] - p[0];
	Vector u2 = p[2] - p[0];

	// compute edge vectors in 2D
	Vector v1 = q[1] - q[0];
	Vector v2 = q[2] - q[0];

	// compute orthonormal basis in 3D
	Vector e1 = u1; e1.normalize();
	Vector e2 = (u2 - dot(u2, e1)*e1); e2.normalize();

	// compute orthonormal basis in 2D
	Vector f1 = v1; f1.normalize();
	Vector f2 = (v2 - dot(v2, f1)*f1); f2.normalize();

	// project onto bases
	p[0] = Vector(0, 0, 0);
	p[1] = Vector(dot(u1, e1), dot(u1, e2), 0);
	p[2] = Vector(dot(u2, e1), dot(u2, e2), 0);

	q[0] = Vector(0, 0, 0);
	q[1] = Vector(dot(v1, f1), dot(v1, f2), 0);
	q[2] = Vector(dot(v2, f1), dot(v2, f2), 0);

        // compute area
	double A = 2.0*cross(u1, u2).norm();

        // compute singular values of mapping from 3D to 2D
	Vector Ss = (q[0]*(p[1].y - p[2].y) + q[1]*(p[2].y - p[0].y) + q[2]*(p[0].y - p[1].y)) / A;
	Vector St = (q[0]*(p[2].x - p[1].x) + q[1]*(p[0].x - p[2].x) + q[2]*(p[1].x - p[0].x)) / A;
	double a = dot(Ss, Ss);
	double b = dot(Ss, St);
	double c = dot(St, St);
	double det = std::sqrt(std::pow(a - c, 2) + 4.0*b*b);
	double Gamma = std::sqrt(0.5*(a + c + det));
	double gamma = std::sqrt(0.5*(a + c - det));

        // reorder so that Gamma is the larger of the two singular values
	if (Gamma < gamma) std::swap(Gamma, gamma);

        // return the ratio of max over min singular value
	return Gamma/gamma;
}

