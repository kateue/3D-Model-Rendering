// C++ include
#include <iostream>
#include <string>
#include <vector>

// Utilities for the Assignment
#include "raster.h"

#include <gif.h>
#include <fstream>

#include <Eigen/Geometry>
// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

using namespace std;
using namespace Eigen;

//Image height
const int H = 480;

//Camera settings
const double near_plane = 1.5; //AKA focal length
const double far_plane = near_plane * 100;
const double field_of_view = 0.7854; //45 degrees
const double aspect_ratio = 1.5;
const bool is_perspective = true;
const Vector3d camera_position(0, 0, 3);
const Vector3d camera_gaze(0, 0, -1);
const Vector3d camera_top(0, 1, 0);

//Object
const std::string data_dir = DATA_DIR;
const std::string mesh_filename(data_dir + "bunny.off");
MatrixXd vertices; // n x 3 matrix (n points)
MatrixXi facets;   // m x 3 matrix (m triangles)

//Material for the object
const Vector3d obj_diffuse_color(0.5, 0.5, 0.5);
const Vector3d obj_specular_color(0.2, 0.2, 0.2);
const double obj_specular_exponent = 256.0;

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector3d> light_colors;
//Ambient light
const Vector3d ambient_light(0.3, 0.3, 0.3);

//Fills the different arrays
void setup_scene()
{
    //Loads file
    std::ifstream in(mesh_filename);
    if (!in.good())
    {
        std::cerr << "Invalid file " << mesh_filename << std::endl;
        exit(1);
    }
    std::string token;
    in >> token;
    int nv, nf, ne;
    in >> nv >> nf >> ne;
    vertices.resize(nv, 3);
    facets.resize(nf, 3);
    for (int i = 0; i < nv; ++i)
    {
        in >> vertices(i, 0) >> vertices(i, 1) >> vertices(i, 2);
    }
    for (int i = 0; i < nf; ++i)
    {
        int s;
        in >> s >> facets(i, 0) >> facets(i, 1) >> facets(i, 2);
        assert(s == 3);
    }

    //Lights
    light_positions.emplace_back(8, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(6, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(4, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(0, 8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-2, -8, 0);
    light_colors.emplace_back(16, 16, 16);

    light_positions.emplace_back(-4, 8, 0);
    light_colors.emplace_back(16, 16, 16);
}

void build_uniform(UniformAttributes &uniform)
{

    double y = (sin(field_of_view / 2) * near_plane / cos(field_of_view / 2));
    double x = (y * aspect_ratio);
    double far_boundary = -far_plane; 
    double near_boundary = -near_plane; 
    double right = x;   
    double left = -x;  
    double top = y;    
    double bottom = -y;  
    
    //TODO: setup uniform

    uniform.view << 1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1;
    
    if (aspect_ratio < 1){

        uniform.view(0, 0) = aspect_ratio;

    }else{

        uniform.view(1, 1) = (1 / aspect_ratio);

    }

    //TODO: setup camera, compute w, u, v
    //TODO: compute the camera transformation

    Vector3d w = (-camera_gaze / camera_gaze.norm());
    Vector3d u = (camera_top.cross(w) / (camera_top.cross(w)).norm());
    Vector3d v = w.cross(u);
    Matrix4d P = Matrix4d(4, 4);

    P.row(0) << u[0], v[0], 
                w[0], camera_position[0];
    P.row(1) << u[1], v[1], 
                w[1], camera_position[1];
    P.row(2) << u[2], v[2], 
                w[2], camera_position[2];
    P.row(3) << 0, 0, 
                0, 1;

    uniform.camera = Matrix4d(4, 4);
    uniform.camera = P.inverse();

    //TODO: setup projection matrix

    uniform.projection.row(0) << 2 / (right - left), 0, 
                                 0, (-(right + left) / (right - left));
    uniform.projection.row(1) << 0, 2 / (top - bottom), 
                                 0, (-(top + bottom) / (top - bottom)); 
    uniform.projection.row(2) << 0, 0, 
                                 2 / (near_boundary - far_boundary), (-(near_boundary + far_boundary) / (near_boundary - far_boundary));
    uniform.projection.row(3) << 0, 0, 
                                 0, 1;

    if (is_perspective){
        //TODO setup prespective camera

        uniform.perspective  << near_boundary, 0, 0, 0,
                                0, near_boundary, 0, 0,
                                0, 0, near_boundary + far_boundary, -far_boundary * near_boundary,
                                0, 0, 1, 0;
    }else{

        uniform.perspective  << 1, 0, 0, 0,
                                0, 1, 0, 0,
                                0, 0, 1, 0,
                                0, 0, 0, 1;

    }
}

void simple_render(Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        
        VertexAttributes result;

        //TODO: fill the shader

        result.position = (uniform.projection * uniform.perspective * uniform.camera * va.position);

        return result;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        
        //TODO: fill the shader

        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        
        //TODO: fill the shader
        
        return FrameBufferAttributes(fa.color[0]*255, fa.color[1]*255, fa.color[2]*255, fa.color[3]*255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    //TODO: build the vertex attributes from vertices and facets

    for(int i = 0; i < facets.rows(); i++){

        VectorXd vertexA = (vertices.row(facets(i, 0)));
        VectorXd vertexB = (vertices.row(facets(i, 1)));
        VectorXd vertexC = (vertices.row(facets(i, 2)));

        vertex_attributes.push_back(vertexA);
        vertex_attributes.push_back(vertexB);
        vertex_attributes.push_back(vertexC);

    }

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

Matrix4d compute_rotation(const double alpha)
{
    //TODO: Compute the rotation matrix of angle alpha on the y axis around the object barycenter
    Matrix4d res;

    if(alpha==0){

        res  << 1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1;

        return res;

    }

    res  << cos(alpha),  0, sin(alpha), 0,
            0,           1, 0,          0,
            -sin(alpha), 0, cos(alpha), 0,
            0,           0, 0,          1;

    
    return res;
}

void wireframe_render(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;

    Matrix4d trafo = compute_rotation(alpha);
    uniform.trafo = trafo;

    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader

        VertexAttributes result;

        result.position = (uniform.projection * uniform.perspective 
                        * uniform.camera * uniform.trafo * va.position);
        
        return result;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: fill the shader

        return FragmentAttributes(1, 0, 0);
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: fill the shader

        return FrameBufferAttributes(fa.color[0]*255, fa.color[1]*255, fa.color[2]*255, fa.color[3]*255);
    };

    std::vector<VertexAttributes> vertex_attributes;

    //TODO: generate the vertex attributes for the edges and rasterize the lines
    //TODO: use the transformation matrix

    for(int i = 0; i < facets.rows(); i++){

        VectorXd vertexA = vertices.row(facets(i, 0));
        VectorXd vertexB = vertices.row(facets(i, 1));
        VectorXd vertexC = vertices.row(facets(i, 2));

        vertex_attributes.push_back(vertexA);
        vertex_attributes.push_back(vertexB);
        vertex_attributes.push_back(vertexB);
        vertex_attributes.push_back(vertexC);
        vertex_attributes.push_back(vertexC);
        vertex_attributes.push_back(vertexA);

    }

    rasterize_lines(program, uniform, vertex_attributes, 0.5, frameBuffer);
}

void get_shading_program(Program &program)
{
    program.VertexShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {

        //TODO: transform the position and the normal
        
        VertexAttributes result;

        result.normal = uniform.trafo * va.normal;
        result.position = uniform.projection * uniform.perspective * 
        uniform.camera * uniform.trafo * va.position;
    
        //TODO: compute the correct lighting
        //TODO: create the correct fragment

        Vector4d lights_color(0, 0, 0, 1);
        const Vector3d norm = Vector3d(result.normal[0], result.normal[1], result.normal[2]);
        const Vector3d pos = Vector3d(result.position[0], result.position[1], result.position[2]);
        
        for (int i = 0; i < light_positions.size(); ++i){
            const Vector3d &li_pos = light_positions[i];
            const Vector4d &li_col = Vector4d(light_colors[i][0], light_colors[i][1], light_colors[i][2], 0);
            const Vector3d light_dir = (li_pos - pos).normalized();

            Vector3d view_dir = (camera_position - pos).normalized(); 
            Vector4d diffuse_col = Vector4d(obj_diffuse_color[0], obj_diffuse_color[1], obj_diffuse_color[2], 0);
            Vector4d specular_col = Vector4d(obj_specular_color[0], obj_specular_color[1], obj_specular_color[2], 0);

            const Vector4d diffuse  = diffuse_col * std::max(light_dir.dot(norm), 0.0);            
            const Vector3d halfway = ((light_dir + view_dir) / (light_dir + view_dir).norm()).normalized();
            const Vector4d specular = specular_col * std::pow(std::max(halfway.dot(norm), 0.0), obj_specular_exponent);
            const Vector3d dist = li_pos - pos;
            
            lights_color = lights_color + (diffuse + specular).cwiseProduct(li_col) / dist.squaredNorm();

        }

        result.color = lights_color;

        result.color[0] = result.color[0] + ambient_light[0];
        result.color[1] = result.color[1] + ambient_light[1];
        result.color[2] = result.color[2] + ambient_light[2];

        return result;
    };

    program.FragmentShader = [](const VertexAttributes &va, const UniformAttributes &uniform) {
        //TODO: create the correct fragment

        FragmentAttributes result(va.color[0], va.color[1], va.color[2], va.color[3]);

        if(is_perspective){

            result.position = Vector4d(va.position[0], va.position[1], va.position[2], va.position[3]);

        }else{

            result.position = Vector4d(va.position[0], va.position[1], -1*va.position[2], va.position[3]);
        
        }

        return result;
    };

    program.BlendingShader = [](const FragmentAttributes &fa, const FrameBufferAttributes &previous) {
        //TODO: implement the depth check

        if(fa.position[2] < previous.depth){

            FrameBufferAttributes result(fa.color[0]*255, fa.color[1]*255, fa.color[2]*255, fa.color[3]*255);

            result.depth = fa.position[2];

            return result;

        }else{


            return previous;

        }
    };
}

void flat_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);
    Eigen::Matrix4d trafo = compute_rotation(alpha);
    uniform.trafo = trafo;

    std::vector<VertexAttributes> vertex_attributes;

    for(int i = 0; i < facets.rows(); i++){

        VectorXd vertexA = vertices.row(facets(i, 0));
        VectorXd vertexB = vertices.row(facets(i, 1));
        VectorXd vertexC = vertices.row(facets(i, 2));

        Vector3d edge1 = vertexB - vertexA;
        Vector3d edge2 = vertexC - vertexA;

        Vector3d facet_norm = edge1.cross(edge2).normalized();
        Vector4d norm  = Vector4d(facet_norm[0], facet_norm[1], facet_norm[2], 0);

        VertexAttributes vertexA_attr = VertexAttributes(vertexA, norm);
        VertexAttributes vertexB_attr = VertexAttributes(vertexB, norm);
        VertexAttributes vertexC_attr = VertexAttributes(vertexC, norm);
        vertex_attributes.push_back(vertexA_attr);
        vertex_attributes.push_back(vertexB_attr);
        vertex_attributes.push_back(vertexC_attr);
    }
    //TODO: compute the normals
    //TODO: set material colors

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

void pv_shading(const double alpha, Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> &frameBuffer)
{
    UniformAttributes uniform;
    build_uniform(uniform);
    Program program;
    get_shading_program(program);
    Eigen::Matrix4d trafo = compute_rotation(alpha);
    uniform.trafo = trafo;

    //TODO: compute the vertex normals as vertex normal average
    std::vector<Vector4d> vertex_normals[vertices.size()];

    for(int i = 0; i < facets.rows(); i++){

        VectorXd vertexA = vertices.row(facets(i, 0));
        VectorXd vertexB = vertices.row(facets(i, 1));
        VectorXd vertexC = vertices.row(facets(i, 2));
        Vector3d edge1 = vertexB - vertexA;
        Vector3d edge2 = vertexC - vertexA;
        Vector3d facet_norm = edge1.cross(edge2).normalized();
        Vector4d norm  = Vector4d(facet_norm[0], facet_norm[1], facet_norm[2], 0);

        vertex_normals[facets(i, 0)].push_back(norm);
        vertex_normals[facets(i, 1)].push_back(norm);
        vertex_normals[facets(i, 2)].push_back(norm);
    }

    std::vector<VertexAttributes> vertex_attributes;

    //TODO: create vertex attributes

    for(int i = 0; i < facets.rows(); i++){

        VectorXd vertexA = vertices.row(facets(i, 0));
        VectorXd vertexB = vertices.row(facets(i, 1));
        VectorXd vertexC = vertices.row(facets(i, 2));
        Vector4d vertex_norm[3];

        for(int j = 0; j < 3; j++){

            for(Vector4d n: vertex_normals[facets(i, j)]){

                vertex_norm[j] = vertex_norm[j] + n;

            }

            vertex_norm[j] = vertex_norm[j] / vertex_normals[facets(i, j)].size();
            vertex_norm[j] = vertex_norm[j].normalized();

        }

        VertexAttributes vertexA_attr = VertexAttributes(vertexA, vertex_norm[0]);
        VertexAttributes vertexB_attr = VertexAttributes(vertexB, vertex_norm[1]);
        VertexAttributes vertexC_attr = VertexAttributes(vertexC, vertex_norm[2]);

        vertex_attributes.push_back(vertexA_attr);
        vertex_attributes.push_back(vertexB_attr);
        vertex_attributes.push_back(vertexC_attr);
    }

    //TODO: set material colors

    rasterize_triangles(program, uniform, vertex_attributes, frameBuffer);
}

int main(int argc, char *argv[])
{
    setup_scene();

    int W = H * aspect_ratio;
    Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic> frameBuffer(W, H);
    vector<uint8_t> image;

    simple_render(frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("simple.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer = Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>::Zero(W,H);
    wireframe_render(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("wireframe.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer = Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>::Zero(W,H);
    flat_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("flat_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    frameBuffer = Eigen::Matrix<FrameBufferAttributes, Eigen::Dynamic, Eigen::Dynamic>::Zero(W,H);
    pv_shading(0, frameBuffer);
    framebuffer_to_uint8(frameBuffer, image);
    stbi_write_png("pv_shading.png", frameBuffer.rows(), frameBuffer.cols(), 4, image.data(), frameBuffer.rows() * 4);

    //TODO: add the animation
    GifWriter g;
    double pi = 3.14159265;
    int delay = 25;
    
    GifBegin(&g, "wireframe.gif", frameBuffer.rows(), frameBuffer.cols(), delay);

    for(double i = 0; i < 1; i += 0.05){
        frameBuffer.setConstant(FrameBufferAttributes());
        wireframe_render(i * 2 * pi, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }

    GifEnd(&g);

    GifBegin(&g, "flat_shading.gif", frameBuffer.rows(), frameBuffer.cols(), delay);

    for(double i = 0; i < 1; i += 0.05){
        frameBuffer.setConstant(FrameBufferAttributes());
        flat_shading(i * 2 * pi, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }

    GifEnd(&g);

    GifBegin(&g, "pv_shading.gif", frameBuffer.rows(), frameBuffer.cols(), delay);

    for(double i = 0; i < 1; i += 0.05){

        frameBuffer.setConstant(FrameBufferAttributes());
        pv_shading(i * 2 * pi, frameBuffer);
        framebuffer_to_uint8(frameBuffer, image);
        GifWriteFrame(&g, image.data(), frameBuffer.rows(), frameBuffer.cols(), delay);
    }
    GifEnd(&g);
    
    return 0;
}