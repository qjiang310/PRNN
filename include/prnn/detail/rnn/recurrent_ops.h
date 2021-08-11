

#pragma once


// Forward Declarations
namespace prnn { class RecurrentOpsHandle; }

namespace prnn { namespace matrix { class DynamicView;      } }
namespace prnn { namespace matrix { class ConstDynamicView; } }
namespace prnn { namespace matrix { class Precision;        } }
namespace prnn { namespace matrix { class Dimension;        } }
namespace prnn { namespace matrix { class Matrix;           } }


namespace prnn
{
namespace rnn
{

size_t getMaximumSizeRNNForThisGPU(const matrix::Precision&);

void forwardPropRecurrent(
    const matrix::DynamicView& activations,
    const matrix::DynamicView& activations_u,
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& weights,
    const matrix::ConstDynamicView& weights_u,
    const matrix::ConstDynamicView& weights_h,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& scratch_u,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle); // ADD INPUTS

void backPropDeltasRecurrent(const matrix::DynamicView& inputDeltas,
    const matrix::ConstDynamicView& weights,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& outputDeltas,
    const matrix::DynamicView& scratch,
    const matrix::DynamicView& reserve,
    const RecurrentOpsHandle& handle);

void backPropGradientsRecurrent(const matrix::DynamicView& dWeights,
    const matrix::ConstDynamicView& inputActivations,
    const matrix::ConstDynamicView& outputActivations,
    const matrix::ConstDynamicView& scratch,
    const matrix::ConstDynamicView& reserve,
    const RecurrentOpsHandle& handle);

size_t getForwardPropScratchSize(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
size_t getBackPropDeltasScratchSize(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
size_t getBackPropGradientsScratchSize(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);

matrix::Dimension getReserveDimensions(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
matrix::Dimension getWeightDimensions(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);

void getWeightsRange(matrix::Dimension& begin, matrix::Dimension& end,
    const RecurrentOpsHandle& handle,
    const matrix::Precision& precision, int index);

matrix::Matrix getForwardPropScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
matrix::Matrix getBackPropDeltasScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);
matrix::Matrix getBackPropGradientsScratch(const RecurrentOpsHandle& handle,
    const matrix::Precision& precision);


}
}


