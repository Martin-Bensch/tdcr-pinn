#include <pybind11/pybind11.h>
#include "../tdcr_modeling/include/tendondrivenrobot.h"
#include "../tdcr_modeling/include/cosseratrodmodel.h"
#include "../tdcr_modeling/include/subsegmentcosseratrodmodel.h"
#include "../tdcr_modeling/include/piecewiseconstantcurvaturemodel.h"
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <Eigen/Core>

namespace py = pybind11;

// Needed to pass python data by reference to Eigen library
void forwardKinematics_pyb_input(const int poses,
                                  py::EigenDRef<const Eigen::VectorXd> q,
                                  py::EigenDRef<const Eigen::VectorXd> f_ext,
                                  py::EigenDRef<const Eigen::VectorXd> l_ext)
{
    // Check q and f length
    if (q.size() % 6 != 0){
        throw std::invalid_argument("Size of q is not a multiple of 6; assuming 3 tendons per segment.");
    }
    if (f_ext.size() != l_ext.size() && f_ext.size() != 3){
        throw std::invalid_argument("f_ext and l_ext are of different size and/or != 3");
    }
    if (q.size() != 6 * poses) {
        throw std::invalid_argument("Size of q does not match to number of poses");
    }

}

// Simple wrapper for the forwardKinematics method. Needed for data storage and
// access.
Eigen::Matrix<bool, Eigen::Dynamic,1> TendonDrivenRobot::forwardKinematics_pyb(
                                 const int poses,
                                 py::EigenDRef<const Eigen::VectorXd> q,
                                 py::EigenDRef<const Eigen::VectorXd> f_ext,
                                 py::EigenDRef<const Eigen::VectorXd> l_ext,
                                 Model model)
{

/* Wraps the forwardKinematics method in TendonDrivon robot for pybind11 */
    try {
        // Check input
        forwardKinematics_pyb_input(poses, q, f_ext, l_ext);

    }catch (const char* message) {
        std::cerr << message << std::endl;
    }
     // Prepare storage for all disk frames
     int number_disks = m_number_disks[0] + m_number_disks[1];

     m_disk_frames_storage.resize(poses  * 4, number_disks * 4 + 4);
     m_disk_states_storage_segment1.resize(poses, m_number_disks[0] * 19 + 19);
     m_disk_states_storage_segment2.resize(poses, m_number_disks[1] * 19);
     m_tendon_displacements_storage.resize(6, poses);
     m_ee_frame_storage.resize(poses * 4, 4);
     m_uv_s0_storage.resize(poses, 6);

    Eigen::Matrix4d ee_frame_tmp;
    Eigen::Matrix<double, 6, 1> q_tmp;

    Eigen::Matrix<bool, Eigen::Dynamic, 1> success_vec;
    success_vec.resize(poses, 1);

    bool success;

    // Run through all given poses
    for (int i = 0; i < poses; i++) {
        // calculate ee, by passing reference to block at (4*i, 0) of size
        // (4,4)
        q_tmp = q.segment(6 * i, 6);
        success = forwardKinematics(
                              ee_frame_tmp,
                              q_tmp,
                              f_ext,
                              l_ext, model);
        // Store success results
        success_vec[i] = success;
        if (success) {
            if (model == TendonDrivenRobot::Model::CosseratRod ||
                model == TendonDrivenRobot::Model::SubsegmentCosseratRod){
                //block(i,j,m,n) start at i,j having m rows and n columns
                m_tendon_displacements_storage.block(0, i, 6, 1) = getTendonDisplacements();

            }
            if (model == TendonDrivenRobot::Model::CosseratRod){
                //block(i,j,m,n) start at i,j having m rows and n columns
                m_tendon_displacements_storage.block(0, i, 6, 1) = getTendonDisplacements();
  //              std:: cout << "mp_cr_model->getStatesSegment1()"<< std::endl;
//                std::cout << mp_cr_model->getStatesSegment1() << std::endl;

                Eigen::MatrixXd states1_tmp = mp_cr_model->getStatesSegment1().transpose();
                states1_tmp.resize(1, m_number_disks[0] * 19 + 19);
  //              std:: cout << "states1 resize tmp"<< std::endl;
        //        std::cout << states1_tmp << std::endl;
                m_disk_states_storage_segment1.block(i,0, 1, m_number_disks[0] * 19 + 19) = states1_tmp;
                //std::cout << "m_disk_states_storage_segment1.block(0,i, 19, 1)" << std::endl;
                //std::cout << m_disk_states_storage_segment1.block(i, 0, 1, m_number_disks[0] * 19 + 19) << std::endl;
               // std::cout << "m_disk_states_storage_segment1" << std::endl;
               // std::cout << m_disk_states_storage_segment1 << std::endl;

               // std:: cout << "mp_cr_model->getStatesSegment2()"<< std::endl;
               // std::cout << mp_cr_model->getStatesSegment2() << std::endl;
                Eigen::MatrixXd states2_tmp = mp_cr_model->getStatesSegment2().transpose();
                states2_tmp.resize(1, m_number_disks[1] * 19);
                //std:: cout << "states2 resize tmp"<< std::endl;
                //std::cout << states2_tmp << std::endl;
                m_disk_states_storage_segment2.block(i, 0, 1, m_number_disks[1] * 19) = states2_tmp;
                m_uv_s0_storage.block(i,0, 1, 6) = getFinalInitValues_uv(model);
            }
            m_disk_frames_storage.block(4 * i, 0, 4, number_disks * 4 + 4) = getDiskFrames();
           /* for (int i; i < m_disk_states_storage_segment1.rows(); i++ ) {


            }*/

            //m_disk_states_storage_segment1.block(4 * i, 0, 4, number_disks * 4 + 4) = getDiskFrames();
            //std::cout << "rows: " << getDiskFrames().rows() << "cols: " <<
            //getDiskFrames().cols() << std::endl;
        }
        m_ee_frame_storage.block(4*i,0, 4,4) = ee_frame_tmp;
    }
    return success_vec;
}


// Eigen::Ref<> wrapper to tendondrivenrobot method in order to call-by
//reference
//Eigen::Ref<MatrixType, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>> same
// as py::EigenDRef<>
const Eigen::VectorXf call_by_reference_eigenref(py::EigenDRef<const Eigen::VectorXf> q)
{
    // Take the given matrix and add 1 to each element:
    int size = q.size();
    Eigen::VectorXf q_1(size);

    for (int i = 0; i < size; i++) {
        q_1(i) = q(i) + 1;
       // std::cout << q(i) << " + 1 =  " << q_1 << std::endl;
    }

    return q_1;
}


CosseratRodModel TendonDrivenRobot::getCosseratRodModel()
{
    // De-reference
    return *mp_cr_model;
}

SubsegmentCosseratRodModel TendonDrivenRobot::getSubsegmentCosseratRodModel()
{
    // De-reference
    return *mp_sscr_model;
}


// Get the initial values for all models which are calculated through optimization
Eigen::MatrixXd TendonDrivenRobot::getDefaultInitValues_uv(Model enum_model)
{
    switch (enum_model) {
        case TendonDrivenRobot::CosseratRod: return mp_cr_model->get_default_inits();
        case TendonDrivenRobot::PiecewiseConstantCurvature: return mp_pcc_model->get_default_inits();
        case TendonDrivenRobot::PseudoRigidBody: return mp_prb_model->get_default_inits();
        case TendonDrivenRobot::SubsegmentCosseratRod: return mp_sscr_model->get_default_inits();
        default: throw std::invalid_argument("f_ext and l_ext are of different size and/or != 3"); break;

    }
}

void TendonDrivenRobot::setDefaultInitValues_uv(Eigen::MatrixXd inits_uv)
{
       mp_cr_model->setDefaultInitValues(inits_uv);
       mp_pcc_model->setDefaultInitValues(inits_uv);
       mp_prb_model->setDefaultInitValues(inits_uv);
       mp_sscr_model->setDefaultInitValues(inits_uv);
}


Eigen::MatrixXd TendonDrivenRobot::getFinalInitValues_uv(Model enum_model)
{
    switch (enum_model) {
        case TendonDrivenRobot::CosseratRod: return mp_cr_model->getFinalInitValues();
        case TendonDrivenRobot::PiecewiseConstantCurvature: return mp_pcc_model->getFinalInitValues();
        case TendonDrivenRobot::PseudoRigidBody: return mp_prb_model->getFinalInitValues();
        case TendonDrivenRobot::SubsegmentCosseratRod: return mp_sscr_model->getFinalInitValues();
        default: throw std::invalid_argument("f_ext and l_ext are of different size and/or != 3"); break;
    }

}

void TendonDrivenRobot::reset_last_inits()
{
    mp_cr_model->reset_last_inits();
    mp_pcc_model->reset_last_inits();
    mp_prb_model->reset_last_inits();
    mp_sscr_model->reset_last_inits();
}

Eigen::MatrixXd TendonDrivenRobot::get_uv_s0_storage()
{
    return m_uv_s0_storage;
}

Eigen::MatrixXd TendonDrivenRobot::getStateSegmentN_CR(int segment)
{
    if (segment == 1)
    {
        return m_disk_states_storage_segment1;

    }else if(segment == 2)
    {
        return m_disk_states_storage_segment2;
    }else
    {
        throw std::invalid_argument("Full states are only provided for the Cosserat rod model.");
    }
}


void TendonDrivenRobot::set_gxf_tol_step_size(float tol, double step_size)
{
    mp_cr_model->set_gxf_tol_step_size(tol, step_size);
}
//Return subsegment cosserat rod model or cosserat rod model
//  ----------------------   Cosserat Rod Model   ------------------------------

Eigen::MatrixXd CosseratRodModel::get_default_inits()
{
    return m_default_inits;
}

void CosseratRodModel::reset_last_inits()
{
    m_last_inits = m_default_inits;
}

Eigen::MatrixXd CosseratRodModel::getStatesSegment1()
{
    return m_states1;
}

Eigen::MatrixXd CosseratRodModel::getStatesSegment2()
{
    return m_states2;
}

void CosseratRodModel::set_gxf_tol_step_size(float tol, double step_size){
    m_gxftol = tol;
    m_step_size = step_size;
}

//  ----------------------  Sub Segment Cosserat Rod Model   ------------------------------
Eigen::MatrixXd SubsegmentCosseratRodModel::get_default_inits()
{
    return m_default_inits;
}

void SubsegmentCosseratRodModel::reset_last_inits()
{
    m_last_inits = m_default_inits;
}
//  ----------------------  Pseudo Rigid Body   ------------------------------

Eigen::MatrixXd PseudoRigidBodyModel::get_default_inits()
{
    return m_default_inits;
}

void PseudoRigidBodyModel::reset_last_inits()
{
    m_last_inits = m_default_inits;
}

//  ----------------------   Piecewise Constant Curvature   ------------------------------

Eigen::MatrixXd PiecewiseConstantCurvatureModel::get_default_inits()
{
    return m_default_inits;
}

void PiecewiseConstantCurvatureModel::reset_last_inits()
{
    m_last_inits = m_default_inits;
}


void tendon_driven_robot_pyb(py::module_ &m) {
    py::class_<TendonDrivenRobot> tdr_ (m, "TendonDrivenRobot");

    tdr_.def(py::init()) //constructor
        .def("setRobotParameters",
              &TendonDrivenRobot::setRobotParameters,
              R"pbdoc(
                    This function allows to set and update the TDCR parameters.

                    Inputs:
                        length:			std::array that holds the length of each of the two segments of the TDCR.
                        youngs_modulus:	Youngs modulus of the backbone of the TDCR.
                        routing:		std::vector that holds the routing position of each tendon of the TDCR expressed
                                        as a 3x1 position vector in the local disk frame. First three entries belong to
                                        the first segment, while the last three entries belong to the second segment.
                        number_disks:	std::array that holds the number of disks for each of the two segments of the
                                        TDCR.
                        pradius_disks:	std::array that holds the pitch radius of the disks (distance between tendon
                                        routing and backbone) for each of the two segments of the TDCR.
                        ro:				Radius of the backbone of the TDCR.
                        two_tendons:	Specifies, if only two tendons for each segment are employed and actuated.
                                        Only affects the implementation of the Constant Curvature modeling approach
                                        (details can be found there).
              )pbdoc",
              py::arg("length"),
              py::arg("youngs_modulus"),
              py::arg("routing"),
              py::arg("number_disks"),
              py::arg("pradius_disks"),
              py::arg("ro"),
              py::arg("two_tendons")
              )
        .def("forwardKinematics_pyb", &TendonDrivenRobot::forwardKinematics_pyb)
       // .def("getTendonDisplacements",&TendonDrivenRobot::getTendonDisplacements)
        .def("getTendonDisplacements_storage", &TendonDrivenRobot::getTendonDisplacements_storage)
        .def("getEEFrames", &TendonDrivenRobot::getEEFrames)
       // .def("getCurrentConfig", &TendonDrivenRobot::getCurrentConfig)
        .def("getDiskFrames_storage", &TendonDrivenRobot::getDiskFrames_storage)
        .def("getStateSegmentN_CR", &TendonDrivenRobot::getStateSegmentN_CR)
        .def("get_uv_s0_storage", &TendonDrivenRobot::get_uv_s0_storage)
        .def("keepInits", &TendonDrivenRobot::keepInits, py::arg("keep"))
        .def("getCosseratRodModel", &TendonDrivenRobot::getCosseratRodModel)
        .def("getSubsegmentCosseratRodModel", &TendonDrivenRobot::getSubsegmentCosseratRodModel)
        .def("getDefaultInitValues_uv", &TendonDrivenRobot::getDefaultInitValues_uv)
        .def("setDefaultInitValues_uv", &TendonDrivenRobot::setDefaultInitValues_uv)
        .def("set_gxf_tol", &TendonDrivenRobot::set_gxf_tol_step_size, py::arg("tol"), py::arg("step_size"))
        .def("getFinalInitValues_uv",
                &TendonDrivenRobot::getFinalInitValues_uv,
                R"pbdoc(
                        This function returns the last values final values of the shooting method for the last solution of the forward kinematics.
	                    It returns 6x1 vector containing the last values for u and v at s = 0.
	                    )pbdoc",
	           py::arg("RobotModel")
	           )
	    .def("reset_last_inits", &TendonDrivenRobot::reset_last_inits
	            );

    // supply a TendonDrivenRobot instance tdr_so that the enumeration is
    // created within the classes scope.
     py::enum_<TendonDrivenRobot::Model>(tdr_, "Model")
        .value("CosseratRod", TendonDrivenRobot::Model::CosseratRod)
        .value("ConstantCurvature",
                TendonDrivenRobot::Model::ConstantCurvature)
        .value("PiecewiseConstantCurvature",
                TendonDrivenRobot::Model::PiecewiseConstantCurvature)
        .value("PseudoRigidBody", TendonDrivenRobot::Model::PseudoRigidBody)
        .value("SubsegmentCosseratRod",
                TendonDrivenRobot::Model::SubsegmentCosseratRod)
        .export_values();
    /*
        .def("setName", &Robot::setName)
        .def("getName", &Robot::getName)
        .def("__repr__",
                [](const Robot &a)
                    {return "<example.Robot named '" + a.getName() + "'>";
                }
             )
        // Access public variables without get/set methods
        .def_readwrite("name_public", &Robot::name_public)
        // Access private c++ class variables
        .def_property("name", &Robot::getName, &Robot::setName);*/
}


void testing(py::module_ &m) {
    m.def("call_by_reference_eigenref", //function name in python
           &call_by_reference_eigenref, //reference to function
           "Call by reference using Eigen::Ref<>", // Function docstring in python
           py::arg("q")); // Define Keywordarguments and
                         //default values
}



// Collect sub-module specific classes
void cosserat_rod_model_pyb(py::module_ &m) {
    py::class_<CosseratRodModel> cr_ (m, "CosseratRodModel");

    cr_.def(py::init()) // Bind constructor
        .def("setDefaultInitValues",
               &CosseratRodModel::setDefaultInitValues,
               R"pbdoc(
               This function allows to set the default initial values that are used as an initial guess for the
               implemented shooting method.

	            Args:
	                inits: 6x1 vector containing the default values that are used as an initial guess for u and v at
	                       s = 0, if continuation mode is disabled.
               )pbdoc",
               py::arg("inits")
               )
        .def("getFinalInitValues",
               &CosseratRodModel::getFinalInitValues,
               R"pbdoc(
                        This function returns the last values final values of the shooting method for the last
                        solution of the forward kinematics. It returns 6x1 vector containing the last values for
                        u and v at s = 0.
               )pbdoc"
               )
        .def("setKeepInits",
               &CosseratRodModel::setKeepInits,
               R"pbdoc(
               This function enables a continuation mode.

	           That means, that every new run of the forward kinematics will use the final values for these variables
	           obtained from the last forward kinematics solution as the new initial guess. This makes sense in cases,
	           where configurations of the robot only change marginally (thus the initial guesses would be similar),
	           and can increase computation time a lot, since the algorithm will converge in just a couple of
	           iterations.

	            Args:
	                keep: Boolean value to indicate, if the continuation mode is used or not.
					      If continuation mode is disabled, the default initial values are used for the initial guess
					      (by default this is the straight, not bent state of the robot, but they can also be changed -
					       see below).
                    )pbdoc",
               py::arg("keep")
               );
}


void subsegmentcosserat_rod_model_pyb(py::module_ &m) {
    py::class_<SubsegmentCosseratRodModel> cr_ (m, "SubsegmentCosseratRodModel");

    cr_.def(py::init()) // Bind constructor
        .def("setDefaultInitValues",
               &SubsegmentCosseratRodModel::setDefaultInitValues,
               R"pbdoc(
               This function allows to set the default initial values that are used as an initial guess for the
               implemented shooting method.

	            Args:
	                inits: 6x1 vector containing the default values that are used as an initial guess for u and v at
	                       s = 0, if continuation mode is disabled.
               )pbdoc",
               py::arg("inits")
               )
        .def("getFinalInitValues",
               &SubsegmentCosseratRodModel::getFinalInitValues,
               R"pbdoc(
                        This function returns the last values final values of the shooting method for the last
                        solution of the forward kinematics. It returns 6x1 vector containing the last values for
                        u and v at s = 0.
               )pbdoc"
               )
        .def("setKeepInits",
               &SubsegmentCosseratRodModel::setKeepInits,
               R"pbdoc(
               This function enables a continuation mode.

	           That means, that every new run of the forward kinematics will use the final values for these variables
	           obtained from the last forward kinematics solution as the new initial guess. This makes sense in cases,
	           where configurations of the robot only change marginally (thus the initial guesses would be similar),
	           and can increase computation time a lot, since the algorithm will converge in just a couple of
	           iterations.

	            Args:
	                keep: Boolean value to indicate, if the continuation mode is used or not.
					      If continuation mode is disabled, the default initial values are used for the initial guess
					      (by default this is the straight, not bent state of the robot, but they can also be changed -
					       see below).
               )pbdoc",
               py::arg("keep")
               );

}

