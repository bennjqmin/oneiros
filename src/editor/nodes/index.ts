// Import all node modules to trigger their registerNode() side-effects.
// FlowEditor imports this file before calling getXYFlowNodeTypes().

// Core
import './InputNode'
import './DenseNode'
import './OutputNode'

// Convolutional
import './Conv2dNode'
import './Conv1dNode'
import './MaxPool2dNode'
import './AvgPool2dNode'
import './AdaptiveAvgPool2dNode'

// Normalisation / regularisation
import './BatchNormNode'
import './DropoutNode'
import './FlattenNode'

// Recurrent
import './RNNNode'
import './LSTMNode'
import './GRUNode'

// Attention / Transformer
import './TransformerEncoderNode'

// Standalone activation
import './ActivationNode'

// Computer Vision
import './ConvTranspose2dNode'
import './UpsampleNode'
import './BackboneNode'
import './HFModelNode'

// Utility / Transformer building blocks
import './GaussianNoiseNode'
import './ReshapeNode'
import './LayerNormNode'
import './EmbeddingNode'
import './PositionalEncodingNode'
import './FeedForwardNode'
import './MultiHeadAttentionNode'

// Merge / multi-input utility
import './ConcatenateNode'
import './AddNode'
import './MultiplyNode'
