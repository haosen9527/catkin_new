#include <iostream>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/core/kernels/smooth-hinge-loss.h>
#include <tensorflow/cc/ops/standard_ops.h>


using namespace std;
using namespace tensorflow;
vector<tensorflow::Output> op_test()
{
    tensorflow::Tensor abslist = tensorflow::Tensor(tensorflow::DT_FLOAT,{8});
    abslist.flat<float>().setRandom();


    tensorflow::Scope root = tensorflow::Scope::NewRootScope();
    tensorflow::ClientSession session(root);

    auto abs_test = tensorflow::ops::Abs(root,abslist);
    //auto assign_abs = tensorflow::ops::Assign(root,tensorflow::ops::Abs(root,abslist),abslist);

    vector<tensorflow::Output> outputs;
    TF_CHECK_OK(session.Run({abs_test},&outputs));
    return outputs;
}



int main()
{
    op_test();
    return 0;
}
