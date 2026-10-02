import torch

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus using PyTorch for pair counting.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """

    # Convert the corpus into token tuples
    words = {}

    for word, frequency in corpus.items():
        words[tuple(word.split())] = frequency

    merges = []

    # Perform BPE merges
    for _ in range(num_merges):

        # Count adjacent pairs
        pair_counts = {}

        for tokens, frequency in words.items():

            for i in range(len(tokens) - 1):

                pair = (tokens[i], tokens[i + 1])

                pair_counts[pair] = pair_counts.get(pair, 0) + frequency

        # Stop if there are no pairs left
        if not pair_counts:
            break

        # Find the pair with the highest frequency
        best_pair = max(pair_counts, key=pair_counts.get)

        # Record the merge
        merges.append(best_pair)

        # Merge the best pair everywhere
        new_words = {}

        for tokens, frequency in words.items():

            new_tokens = []
            i = 0

            while i < len(tokens):

                if (
                    i < len(tokens) - 1
                    and tokens[i] == best_pair[0]
                    and tokens[i + 1] == best_pair[1]
                ):
                    # Merge two tokens into one
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2

                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_words[tuple(new_tokens)] = frequency

        words = new_words

    return merges