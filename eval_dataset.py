"""
Benchmark evaluation dataset for Mini RAG.
Curated directly from 'Artificial Intelligence, Machine Learning, and Deep Learning.pdf'.
"""

BENCHMARK_DATASET = [
    {
        "id": "Q1",
        "category": "AI Foundations",
        "question": "What is the Turing Test?",
        "expected_pages": [24, 25],
        "ground_truth_answer": (
            "The Turing Test was proposed by Alan Turing to determine whether a machine exhibits intelligence. "
            "It involves an imitation game where an interrogator is separated by a curtain from a man and a woman, "
            "and asks questions to determine which is which. In the machine context, the machine attempts to imitate "
            "a human to convince the interrogator of its intelligence."
        ),
        "key_terms": ["Turing", "imitation game", "interrogator", "intelligence"],
        "is_out_of_scope": False
    },
    {
        "id": "Q2",
        "category": "AI Foundations",
        "question": "What is the difference between Strong AI and Weak AI?",
        "expected_pages": [23, 24],
        "ground_truth_answer": (
            "Weak AI views any system that exhibits intelligent behavior as an example of AI, focusing on practical problem solving. "
            "In contrast, Strong AI asserts that computers can actually possess consciousness, sentience, and genuine minds."
        ),
        "key_terms": ["weak AI", "strong AI", "consciousness", "behavior"],
        "is_out_of_scope": False
    },
    {
        "id": "Q3",
        "category": "Machine Learning Classifiers",
        "question": "What is the kNN algorithm?",
        "expected_pages": [86, 87],
        "ground_truth_answer": (
            "The kNN (k-Nearest Neighbors) algorithm is a classification algorithm where data points that are near "
            "each other are classified as belonging to the same class. When a new point is introduced, it is assigned to "
            "the class of the majority of its k nearest neighbors by majority vote."
        ),
        "key_terms": ["k Nearest Neighbor", "classification", "majority", "nearest"],
        "is_out_of_scope": False
    },
    {
        "id": "Q4",
        "category": "Deep Learning",
        "question": "What is a Perceptron and how is its function defined?",
        "expected_pages": [122, 123],
        "ground_truth_answer": (
            "A Perceptron is an artificial neuron with incoming edges that have numeric weights. "
            "Its function f(x) is defined as f(x) = 1 if w*x + b > 0, otherwise f(x) = 0, "
            "where w is a vector of weights, x is an input vector, and b is the bias."
        ),
        "key_terms": ["Perceptron", "weights", "bias", "f(x)"],
        "is_out_of_scope": False
    },
    {
        "id": "Q5",
        "category": "Deep Learning Architectures",
        "question": "What is BPTT in RNNs?",
        "expected_pages": [149],
        "ground_truth_answer": (
            "BPTT stands for Backpropagation Through Time. In Recurrent Neural Networks (RNNs), BPTT is the counterpart "
            "to standard backpropagation for CNNs, where the weight matrices of RNNs are updated during training across time steps."
        ),
        "key_terms": ["BPTT", "Backpropagation Through Time", "RNN", "weight"],
        "is_out_of_scope": False
    },
    {
        "id": "Q6",
        "category": "Deep Learning Architectures",
        "question": "What is the role of the Max Pooling Layer in CNNs?",
        "expected_pages": [134, 135],
        "ground_truth_answer": (
            "Max pooling in CNNs is performed after processing the feature map with an activation function like ReLU. "
            "It reduces the spatial dimensions of the feature map by selecting the maximum value within each pooling window, "
            "decreasing computation and helping achieve spatial invariance."
        ),
        "key_terms": ["max pooling", "feature map", "ReLU", "maximum"],
        "is_out_of_scope": False
    },
    {
        "id": "Q7",
        "category": "Machine Learning Preprocessing",
        "question": "What is data normalization and how is it calculated?",
        "expected_pages": [54],
        "ground_truth_answer": (
            "Data normalization is a linear scaling technique. For a dataset with values Xi, where Minx is the minimum "
            "value and Maxx is the maximum value, the new normalized value is calculated as: Xi = (Xi - Minx) / (Maxx - Minx)."
        ),
        "key_terms": ["normalization", "linear scaling", "Minx", "Maxx"],
        "is_out_of_scope": False
    },
    {
        "id": "Q8",
        "category": "Neural Networks",
        "question": "What are common activation functions used in neural networks?",
        "expected_pages": [10, 80, 81, 82, 83, 84, 85, 86],
        "ground_truth_answer": (
            "Common activation functions include ReLU (Rectified Linear Unit), ELU, Sigmoid, Softmax, Softplus, and Tanh."
        ),
        "key_terms": ["ReLU", "Sigmoid", "Softmax", "activation function"],
        "is_out_of_scope": False
    },
    {
        "id": "Q9",
        "category": "Out-of-Scope (Hallucination Test)",
        "question": "What does the text say about quantum gravity and string theory?",
        "expected_pages": [],
        "ground_truth_answer": "The provided context does not contain information about quantum gravity or string theory.",
        "key_terms": ["not", "context"],
        "is_out_of_scope": True
    },
    {
        "id": "Q10",
        "category": "Out-of-Scope (Hallucination Test)",
        "question": "Who won the FIFA World Cup in 2022?",
        "expected_pages": [],
        "ground_truth_answer": "The provided context does not contain information about the 2022 FIFA World Cup.",
        "key_terms": ["not", "context"],
        "is_out_of_scope": True
    }
]
