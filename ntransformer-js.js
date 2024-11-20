// Utility functions
const exists = (v) => v !== null && v !== undefined;
const defaultTo = (v, d) => exists(v) ? v : d;

// L2 normalization function
const l2norm = (tensor, dim = -1, normEps = 0, groups = 1) => {
    // Simple implementation for vector normalization
    const norm = Math.sqrt(tensor.reduce((acc, val) => acc + val * val, 0));
    return tensor.map(val => val / (norm + 1e-6));
};

class Scale {
    constructor(dim, init = 1.0, scale = 1.0) {
        this.scale = new Array(dim).fill(scale);
        this.forwardScale = init / scale;
    }

    forward() {
        return this.scale.map(s => s * this.forwardScale);
    }
}

class Attention {
    constructor({
        dim,
        dimHead = 64,
        heads = 8,
        normQk = true,
        causal = true,
        sQkInit = 1.0,
        sQkScale = null,
        normEps = 0,
        numHyperspheres = 1
    }) {
        this.heads = heads;
        this.causal = causal;
        this.dimHead = dimHead;
        this.normQk = normQk;
        this.attnScale = Math.sqrt(dimHead);
        
        const dimInner = dimHead * heads;
        
        // Initialize weight matrices
        this.Wq = this.initializeWeight(dim, dimInner);
        this.Wk = this.initializeWeight(dim, dimInner);
        this.Wv = this.initializeWeight(dim, dimInner);
        this.Wo = this.initializeWeight(dimInner, dim);
        
        this.qkScale = new Scale(dimInner, sQkInit, defaultTo(sQkScale, 1/Math.sqrt(dim)));
    }

    initializeWeight(inDim, outDim) {
        // Initialize with random values scaled by sqrt(1/inDim)
        const scale = Math.sqrt(1/inDim);
        return Array(outDim).fill().map(() => 
            Array(inDim).fill().map(() => (Math.random() - 0.5) * 2 * scale)
        );
    }

    matMul(a, b) {
        const result = Array(a.length).fill(0);
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b.length; j++) {
                result[i] += a[i] * b[j];
            }
        }
        return result;
    }

    splitHeads(x) {
        // Reshape input tensor into [batch, heads, seq_len, dim_head]
        // Simplified implementation for a single sequence
        const batchSize = 1;
        const seqLen = x.length / (this.heads * this.dimHead);
        
        const result = [];
        for (let h = 0; h < this.heads; h++) {
            const head = [];
            for (let s = 0; s < seqLen; s++) {
                const start = h * this.dimHead + s * this.heads * this.dimHead;
                head.push(x.slice(start, start + this.dimHead));
            }
            result.push(head);
        }
        return result;
    }

    mergeHeads(x) {
        // Reshape [batch, heads, seq_len, dim_head] back to [batch, seq_len, heads * dim_head]
        const result = [];
        const seqLen = x[0].length;
        
        for (let s = 0; s < seqLen; s++) {
            for (let h = 0; h < this.heads; h++) {
                result.push(...x[h][s]);
            }
        }
        return result;
    }

    forward(x, mask = null) {
        // Project input to q, k, v
        const q = this.matMul(this.Wq, x);
        const k = this.matMul(this.Wk, x);
        const v = this.matMul(this.Wv, x);

        // Split heads
        let qh = this.splitHeads(q);
        let kh = this.splitHeads(k);
        let vh = this.splitHeads(v);

        // Apply normalization if needed
        if (this.normQk) {
            qh = qh.map(head => head.map(seq => l2norm(seq)));
            kh = kh.map(head => head.map(seq => l2norm(seq)));
        }

        // Scale q and k
        const qkScale = this.qkScale.forward();
        qh = qh.map(head => head.map(seq => seq.map(x => x * qkScale[0])));

        // Compute attention scores
        const scores = qh.map((headQ, h) => {
            const headScores = [];
            for (let i = 0; i < headQ.length; i++) {
                const rowScores = [];
                for (let j = 0; j < kh[h].length; j++) {
                    if (this.causal && j > i) {
                        rowScores.push(-Infinity);
                    } else {
                        let score = 0;
                        for (let d = 0; d < this.dimHead; d++) {
                            score += headQ[i][d] * kh[h][j][d];
                        }
                        rowScores.push(score / this.attnScale);
                    }
                }
                headScores.push(rowScores);
            }
            return headScores;
        });

        // Apply softmax
        const attnWeights = scores.map(headScores => 
            headScores.map(rowScores => {
                const maxScore = Math.max(...rowScores);
                const expScores = rowScores.map(s => Math.exp(s - maxScore));
                const sumExp = expScores.reduce((a, b) => a + b, 0);
                return expScores.map(s => s / sumExp);
            })
        );

        // Apply attention weights to values
        const attnOut = attnWeights.map((headWeights, h) => 
            headWeights.map(rowWeights => {
                const weightedValues = new Array(this.dimHead).fill(0);
                for (let j = 0; j < rowWeights.length; j++) {
                    for (let d = 0; d < this.dimHead; d++) {
                        weightedValues[d] += rowWeights[j] * vh[h][j][d];
                    }
                }
                return weightedValues;
            })
        );

        // Merge heads and project output
        const merged = this.mergeHeads(attnOut);
        return this.matMul(this.Wo, merged);
    }
}

class FeedForward {
    constructor({
        dim,
        expandFactor = 4,
        sHiddenInit = 1.0,
        sHiddenScale = 1.0,
        sGateInit = 1.0,
        sGateScale = 1.0,
        normEps = 0,
        numHyperspheres = 1
    }) {
        this.dim = dim;
        const dimInner = Math.floor(dim * expandFactor * 2/3);

        // Initialize weights
        this.Wu = this.initializeWeight(dim, dimInner);
        this.Wv = this.initializeWeight(dim, dimInner);
        this.Wo = this.initializeWeight(dimInner, dim);

        this.hiddenScale = new Scale(dimInner, sHiddenInit, sHiddenScale);
        this.gateScale = new Scale(dimInner, sGateInit, sGateScale);
    }

    initializeWeight(inDim, outDim) {
        const scale = Math.sqrt(1/inDim);
        return Array(outDim).fill().map(() => 
            Array(inDim).fill().map(() => (Math.random() - 0.5) * 2 * scale)
        );
    }

    silu(x) {
        return x * (1 / (1 + Math.exp(-x)));
    }

    matMul(a, b) {
        const result = Array(a.length).fill(0);
        for (let i = 0; i < a.length; i++) {
            for (let j = 0; j < b.length; j++) {
                result[i] += a[i] * b[j];
            }
        }
        return result;
    }

    forward(x) {
        const u = this.matMul(this.Wu, x);
        const v = this.matMul(this.Wv, x);

        const hiddenScale = this.hiddenScale.forward();
        const gateScale = this.gateScale.forward();

        const hidden = u.map((val, i) => val * hiddenScale[i]);
        const gate = v.map((val, i) => val * gateScale[i] * Math.sqrt(this.dim));

        const activated = hidden.map((h, i) => h * this.silu(gate[i]));
        return this.matMul(this.Wo, activated);
    }
}

class nTransformer {
    constructor({
        dim,
        depth,
        dimHead = 64,
        heads = 8,
        attnNormQk = true,
        ffExpandFactor = 4,
        alphaInit = null,
        normEps = 0
    }) {
        this.dim = dim;
        this.layers = [];
        alphaInit = defaultTo(alphaInit, 1 / depth);

        for (let i = 0; i < depth; i++) {
            const attn = new Attention({
                dim,
                dimHead,
                heads,
                normQk: attnNormQk,
                sQkInit: 1.0,
                normEps
            });

            const ff = new FeedForward({
                dim,
                expandFactor: ffExpandFactor,
                normEps
            });

            const attnInterp = new Scale(dim, alphaInit, 1/Math.sqrt(dim));
            const ffInterp = new Scale(dim, alphaInit, 1/Math.sqrt(dim));

            this.layers.push([attn, ff, attnInterp, ffInterp]);
        }
    }

    forward(tokens, mask = null, normInput = false) {
        let x = [...tokens];
        
        if (normInput) {
            x = l2norm(x);
        }

        for (const [attn, ff, attnAlpha, ffAlpha] of this.layers) {
            // Attention block
            const attnOut = l2norm(attn.forward(x, mask));
            const attnScale = attnAlpha.forward();
            x = l2norm(x.map((xi, i) => xi + attnScale[i] * (attnOut[i] - xi)));

            // FeedForward block
            const ffOut = l2norm(ff.forward(x));
            const ffScale = ffAlpha.forward();
            x = l2norm(x.map((xi, i) => xi + ffScale[i] * (ffOut[i] - xi)));
        }

        return x;
    }
}

class Trainer {
    constructor(model, config = {}) {
        this.model = model;
        this.learningRate = config.learningRate || 0.001;
        this.batchSize = config.batchSize || 32;
        this.epochs = config.epochs || 10;
        this.clipGradient = config.clipGradient || 1.0;
    }

    // Calculate loss (using cross-entropy loss as an example)
    calculateLoss(predictions, targets) {
        let loss = 0;
        for (let i = 0; i < predictions.length; i++) {
            loss -= targets[i] * Math.log(predictions[i] + 1e-9);
        }
        return loss;
    }

    // Simple gradient calculation (using finite differences)
    calculateGradients(params, loss, epsilon = 1e-7) {
        const gradients = [];
        for (let i = 0; i < params.length; i++) {
            const original = params[i];
            params[i] = original + epsilon;
            const lossPlus = this.calculateLoss(...arguments);
            params[i] = original - epsilon;
            const lossMinus = this.calculateLoss(...arguments);
            params[i] = original;
            
            const gradient = (lossPlus - lossMinus) / (2 * epsilon);
            gradients.push(gradient);
        }
        return gradients;
    }

    // Update model parameters using gradients
    updateParameters(params, gradients) {
        for (let i = 0; i < params.length; i++) {
            // Clip gradients to prevent explosion
            const clippedGradient = Math.max(
                Math.min(gradients[i], this.clipGradient),
                -this.clipGradient
            );
            params[i] -= this.learningRate * clippedGradient;
        }
    }

    // Training step for a single batch
    trainStep(batch, targets) {
        let totalLoss = 0;

        for (let i = 0; i < batch.length; i++) {
            // Forward pass
            const predictions = this.model.forward(batch[i], null, true);
            
            // Calculate loss
            const loss = this.calculateLoss(predictions, targets[i]);
            totalLoss += loss;

            // Calculate gradients and update parameters
            // Note: This is a simplified version - in practice, you'd want to use
            // automatic differentiation for more efficient gradient calculation
            for (const layer of this.model.layers) {
                const [attn, ff, attnAlpha, ffAlpha] = layer;
                
                // Update attention parameters
                this.updateParameters(attn.Wq.flat(), this.calculateGradients(attn.Wq.flat(), loss));
                this.updateParameters(attn.Wk.flat(), this.calculateGradients(attn.Wk.flat(), loss));
                this.updateParameters(attn.Wv.flat(), this.calculateGradients(attn.Wv.flat(), loss));
                this.updateParameters(attn.Wo.flat(), this.calculateGradients(attn.Wo.flat(), loss));
                
                // Update feedforward parameters
                this.updateParameters(ff.Wu.flat(), this.calculateGradients(ff.Wu.flat(), loss));
                this.updateParameters(ff.Wv.flat(), this.calculateGradients(ff.Wv.flat(), loss));
                this.updateParameters(ff.Wo.flat(), this.calculateGradients(ff.Wo.flat(), loss));
            }
        }

        return totalLoss / batch.length;
    }

    // Main training loop
    async train(dataset, validationSet = null) {
        const trainHistory = {
            losses: [],
            validationLosses: []
        };

        for (let epoch = 0; epoch < this.epochs; epoch++) {
            let epochLoss = 0;
            let batchCount = 0;

            // Split dataset into batches
            for (let i = 0; i < dataset.inputs.length; i += this.batchSize) {
                const batchInputs = dataset.inputs.slice(i, i + this.batchSize);
                const batchTargets = dataset.targets.slice(i, i + this.batchSize);

                const batchLoss = this.trainStep(batchInputs, batchTargets);
                epochLoss += batchLoss;
                batchCount++;

                // Log progress
                if (batchCount % 10 === 0) {
                    console.log(`Epoch ${epoch + 1}/${this.epochs}, Batch ${batchCount}, Loss: ${batchLoss.toFixed(4)}`);
                }
            }

            const averageEpochLoss = epochLoss / batchCount;
            trainHistory.losses.push(averageEpochLoss);

            // Validation step
            if (validationSet) {
                const validationLoss = this.validate(validationSet);
                trainHistory.validationLosses.push(validationLoss);
                console.log(`Epoch ${epoch + 1} completed. Training Loss: ${averageEpochLoss.toFixed(4)}, Validation Loss: ${validationLoss.toFixed(4)}`);
            } else {
                console.log(`Epoch ${epoch + 1} completed. Training Loss: ${averageEpochLoss.toFixed(4)}`);
            }
        }

        return trainHistory;
    }

    // Validation step
    validate(validationSet) {
        let totalLoss = 0;
        
        for (let i = 0; i < validationSet.inputs.length; i++) {
            const predictions = this.model.forward(validationSet.inputs[i], null, true);
            totalLoss += this.calculateLoss(predictions, validationSet.targets[i]);
        }

        return totalLoss / validationSet.inputs.length;
    }
}

// Example usage
function trainExample() {
    // Create model
    const model = new nTransformer({
        dim: 64,
        depth: 2,
        dimHead: 32,
        heads: 2
    });

    // Create trainer
    const trainer = new Trainer(model, {
        learningRate: 0.001,
        batchSize: 16,
        epochs: 5
    });

    // Create dummy dataset
    const dataset = {
        inputs: Array(100).fill().map(() => 
            Array(64).fill().map(() => Math.random() - 0.5)
        ),
        targets: Array(100).fill().map(() => 
            Array(64).fill().map(() => Math.random())
        )
    };

    // Train model
    trainer.train(dataset).then(history => {
        console.log('Training completed');
        console.log('Final loss:', history.losses[history.losses.length - 1]);
    });
}

// Test function
function test() {
    // Create a small transformer
    const model = new nTransformer({
        dim: 64,
        depth: 2,
        dimHead: 32,
        heads: 2
    });

    // Create random input
    const input = Array(64).fill().map(() => Math.random() - 0.5);

    console.time('Forward Pass');
    const output = model.forward(input, null, true);
    console.timeEnd('Forward Pass');

    console.log('Input shape:', input.length);
    console.log('Output shape:', output.length);
    console.log('First few output values:', output.slice(0, 5));
    
    // Test L2 norm
    const inputNorm = Math.sqrt(input.reduce((acc, val) => acc + val * val, 0));
    const outputNorm = Math.sqrt(output.reduce((acc, val) => acc + val * val, 0));
    
    console.log('Input L2 norm:', inputNorm);
    console.log('Output L2 norm:', outputNorm);
    console.log('Normalized as expected:', Math.abs(outputNorm - 1) < 1e-6);
}
