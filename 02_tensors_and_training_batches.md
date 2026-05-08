# Lesson 02: Tensors And Training Batches

Lesson 01 explained the mental model:

```text
text -> token ids -> embeddings -> causal attention -> MLP -> logits -> next token
```

This lesson turns that idea into the shapes a decoder-only transformer actually
uses.

The goal is not to build the whole model yet. The goal is to understand the data
that enters the model and the data that comes out.

By the end, this should feel natural:

```text
input ids:  [batch, time]
targets:    [batch, time]
embeddings: [batch, time, channels]
logits:     [batch, time, vocab_size]
```

## Step 1: Learn The Three Shape Words

We will use three shape names constantly:

```text
B = batch size
T = sequence length, also called time
C = channel size, also called embedding size
```

If:

```text
B = 2
T = 4
C = 8
```

that means:

```text
2 sequences in the batch
4 token positions per sequence
8 numbers in each token vector
```

Decoder-only transformer code becomes much easier when you can read shapes like
a sentence.

```text
[B, T]    means one token id per batch item and time position
[B, T, C] means one vector per batch item and time position
```

## Step 2: Start With A Tiny Token Dataset

Use the same tiny vocabulary idea from Lesson 01:

```text
0 = <pad>
1 = the
2 = cat
3 = sat
4 = on
5 = mat
```

Imagine we have one long stream of token ids:

```text
[1, 2, 3, 4, 1, 5, 1, 2, 3]
```

As words:

```text
the cat sat on the mat the cat sat
```

Language model training usually samples small chunks from a longer token stream.
The model does not train on one sentence only. It sees many short windows from
large text.

## Step 3: Choose A Context Length

The context length is how many token positions the model sees at once.

In this lesson, use:

```text
T = 4
```

A training window might be:

```text
[1, 2, 3, 4, 1]
```

This has 5 tokens, not 4, because we need both inputs and next-token targets.

We split it like this:

```text
input ids:  [1, 2, 3, 4]
target ids: [2, 3, 4, 1]
```

Read it position by position:

```text
see "the"             -> predict "cat"
see "the cat"         -> predict "sat"
see "the cat sat"     -> predict "on"
see "the cat sat on"  -> predict "the"
```

This is why a context of length `T` needs `T + 1` raw tokens to create one
training example.

## Step 4: Make A Batch

Training one sequence at a time is slow. We group multiple examples into a
batch.

Example with:

```text
B = 2
T = 4
```

Two input rows:

```text
[
  [1, 2, 3, 4],
  [5, 1, 2, 3],
]
```

Two target rows:

```text
[
  [2, 3, 4, 1],
  [1, 2, 3, 4],
]
```

Now the shape is:

```text
inputs.shape  = [2, 4]
targets.shape = [2, 4]
```

This is the first real input to the model:

```text
token ids shaped [B, T]
```

Each cell is one integer token id.

## Step 5: Convert Token Ids To Embeddings

The model cannot use token ids directly as meaning. It uses an embedding table.

If:

```text
vocab_size = 6
C = 8
```

then the token embedding table has shape:

```text
[vocab_size, C] = [6, 8]
```

That means:

```text
6 possible token ids
8 learned numbers per token
```

When the model looks up embeddings for inputs shaped `[B, T]`, the result is:

```text
token embeddings shape = [B, T, C]
```

For our tiny batch:

```text
inputs.shape = [2, 4]
```

the embedding output is:

```text
x.shape = [2, 4, 8]
```

Read that as:

```text
2 sequences
4 positions per sequence
8 numbers per position
```

## Step 6: Add Position Embeddings

Token embeddings know what token is present.

Position embeddings tell the model where the token is.

If:

```text
T = 4
C = 8
```

then the position embedding table for this context has shape:

```text
[T, C] = [4, 8]
```

One position vector exists for each position:

```text
position 0 -> vector of size 8
position 1 -> vector of size 8
position 2 -> vector of size 8
position 3 -> vector of size 8
```

The model adds token and position embeddings:

```text
x = token_embedding(input_ids) + position_embedding(positions)
```

The shape stays:

```text
x.shape = [B, T, C]
```

Nothing mysterious happened. Each token position now has a vector that contains:

```text
what token this is
where this token is
```

## Step 7: Keep The Same Shape Through Decoder Blocks

Decoder blocks update the vectors, but they usually keep the same shape.

Before a block:

```text
x.shape = [B, T, C]
```

After causal self-attention:

```text
x.shape = [B, T, C]
```

After the MLP:

```text
x.shape = [B, T, C]
```

The meaning changes, not the shape.

At the start, each vector mostly represents token and position information.
After several decoder blocks, each vector represents richer context about the
prefix visible at that position.

For example, the vector at position 3 can contain information from positions:

```text
0, 1, 2, 3
```

But because of the causal mask, the vector at position 1 cannot contain
information from position 2 or 3.

## Step 8: Project Hidden Vectors To Logits

The final hidden vectors are still shaped:

```text
[B, T, C]
```

To predict the next token, the model maps each vector to vocabulary scores.

If:

```text
C = 8
vocab_size = 6
```

the final projection changes the last dimension:

```text
[B, T, C] -> [B, T, vocab_size]
```

For our example:

```text
[2, 4, 8] -> [2, 4, 6]
```

The result is called logits.

```text
logits[b, t] = scores for the next token at batch row b, time position t
```

If `logits[0, 2]` is a vector of 6 scores, those scores correspond to:

```text
score for <pad>
score for the
score for cat
score for sat
score for on
score for mat
```

The target at the same position tells us which score should be high.

## Step 9: Match Logits Against Targets

The shapes are:

```text
logits.shape  = [B, T, vocab_size]
targets.shape = [B, T]
```

For each `[b, t]` position:

```text
logits[b, t] contains one score per vocabulary token
targets[b, t] contains the correct next-token id
```

Example:

```text
input row:  [1, 2, 3, 4]
target row: [2, 3, 4, 1]
```

At position 0:

```text
input token = 1, "the"
target id   = 2, "cat"
```

So the model wants:

```text
logits[0, 0, 2] to be high
```

At position 1:

```text
input prefix = "the cat"
target id    = 3, "sat"
```

So the model wants:

```text
logits[0, 1, 3] to be high
```

The loss checks this for every batch row and every time position.

## Step 10: Know The Full Shape Story

Here is the whole Lesson 02 flow:

```text
raw token stream:
    [many_tokens]

sample windows:
    [B, T + 1]

split:
    inputs  [B, T]
    targets [B, T]

embed:
    token embeddings [B, T, C]
    position added   [B, T, C]

decoder blocks:
    hidden states    [B, T, C]

final projection:
    logits           [B, T, vocab_size]

loss:
    compares logits [B, T, vocab_size]
    against targets [B, T]
```

This is the core training batch for a decoder-only transformer.

## Step 11: Write Your Lesson 02 Notes

Write a short note using this structure:

```markdown
# Lesson 02 Notes

## Shape Symbols

B =
T =
C =
vocab_size =

## One Training Example

raw tokens:
input ids:
target ids:

Explain why the raw tokens need length T + 1.

## Batch Shape

If B = 2 and T = 4:

inputs shape =
targets shape =

## Model Shape Flow

Fill this in:

input ids:
embeddings:
after decoder blocks:
logits:

## One Position

Pick one input position and explain which target token it should predict.
```

Keep this practical. If the shape story is clear, the first PyTorch
implementation becomes much easier.

## Done Checklist

You are done when:

- you can explain why inputs and targets both have shape `[B, T]`
- you can explain why raw sampled chunks need length `T + 1`
- you can describe why embeddings have shape `[B, T, C]`
- you can describe why logits have shape `[B, T, vocab_size]`
- you can point to one `[b, t]` position and name the target token it learns

Stop here. Lesson 03 can start building the tiny training-batch code.
