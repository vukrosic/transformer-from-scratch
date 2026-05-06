# Week 01: Decoder-Only Transformer Mental Model

This week teaches the first habit of building a decoder-only LLM from scratch:

```text
an LLM is trained to predict the next token using only the tokens to its left
```

You will not write model code yet. You will learn what a decoder-only
transformer is responsible for, trace one tiny sequence through the model,
understand causal masking, and write your first plain-English explanation of
the forward pass.

Work through this file from top to bottom. Every task appears inside the lesson
at the moment you need it.

## Step 1: Understand The Job Of Week 01

The goal this week is not to build ChatGPT. The goal is to understand the exact
problem a decoder-only transformer solves.

The workflow is:

```text
1. start with a short text sequence
2. turn the text into token ids
3. turn token ids into vectors
4. mix information with causal self-attention
5. transform each position with an MLP
6. predict the next token at every position
```

For Week 01, the trusted implementation is your mental model. Later lessons
will turn each part into code. Before that, you need to know what each part is
supposed to do.

## Step 2: Learn What Decoder-Only Means

A transformer can be used in different shapes:

- an encoder reads an input sequence and builds representations
- a decoder generates tokens one at a time
- a decoder-only LLM uses only the decoder stack

Models like GPT-style language models are decoder-only. They read a prefix of
tokens and predict what token should come next.

Example:

```text
input prefix:
The cat sat on the

target next token:
mat
```

During training, the model does this for many positions in the same sequence.
If the text is:

```text
the cat sat
```

then the training pairs are:

```text
given "the"       predict "cat"
given "the cat"   predict "sat"
```

This is the first useful decoder-only question:

```text
At this position, which earlier tokens am I allowed to use?
```

The answer must never include future tokens.

## Step 3: Learn The Words Token, Context, Logit, And Loss

Decoder-only LLM work uses four words constantly:

- **token** means a small piece of text represented as an integer id
- **context** means the tokens the model is allowed to look at
- **logit** means an unnormalized score for a possible next token
- **loss** means how wrong the model was before learning from the mistake

For now, keep the model simple:

```text
tokens:
    integers that stand in for text pieces

context:
    all tokens to the left of the prediction point

logits:
    one score per vocabulary item

loss:
    a number that gets smaller when the right next token receives a higher score
```

You are not implementing loss this week. You only need to see where the model's
prediction comes from.

## Step 4: Trace A Tiny Training Example

Use this toy vocabulary:

```text
0 = <pad>
1 = the
2 = cat
3 = sat
4 = mat
```

Now use this text:

```text
the cat sat
```

The token ids are:

```text
[1, 2, 3]
```

Training shifts the sequence by one position:

```text
input ids:  [1, 2]
target ids: [2, 3]
```

Read that as:

```text
position 0 sees "the" and should predict "cat"
position 1 sees "the cat" and should predict "sat"
```

This shift is the heart of next-token prediction. The model is not given a
separate label file. The text itself creates the labels by shifting one token to
the right.

## Step 5: Add Embeddings And Positions

Token ids are not enough for a neural network. The model turns each id into a
vector.

```text
token id 1 -> embedding for "the"
token id 2 -> embedding for "cat"
```

The model also needs position information. Without positions, the sequence:

```text
the cat
```

would look too similar to:

```text
cat the
```

So the first representation is:

```text
input vector at position 0 = token embedding("the") + position embedding(0)
input vector at position 1 = token embedding("cat") + position embedding(1)
```

This creates one vector per token position. The rest of the transformer updates
those vectors.

## Step 6: Learn Causal Self-Attention

Self-attention lets each token position gather information from other token
positions. In a decoder-only LLM, the attention is causal.

Causal means:

```text
a position can look left
a position can look at itself
a position cannot look right
```

For the sequence:

```text
[the, cat, sat]
```

the allowed attention pattern is:

```text
position 0 ("the") can attend to:
    the

position 1 ("cat") can attend to:
    the, cat

position 2 ("sat") can attend to:
    the, cat, sat
```

The blocked pattern is:

```text
position 0 cannot attend to "cat" or "sat"
position 1 cannot attend to "sat"
```

This is what prevents cheating during training. If the model saw the future
token, next-token prediction would become a lookup instead of learning.

## Step 7: Trace One Decoder Block

A decoder block has two main sublayers:

```text
1. causal self-attention
2. MLP
```

It also uses residual connections:

```text
x = x + causal_attention(x)
x = x + mlp(x)
```

Read this as:

```text
start with the current token vectors
mix context using causal attention
add that result back to the original stream
transform each position with an MLP
add that result back too
```

The residual stream is the running workspace of the model. Each block edits the
workspace, but the original signal can still flow forward.

## Step 8: Produce The Next-Token Scores

After the final decoder block, the model has one vector per position.

Each vector is projected into vocabulary scores:

```text
vector at position 0 -> logits for next token after "the"
vector at position 1 -> logits for next token after "the cat"
```

If the vocabulary has 5 tokens, each position produces 5 logits:

```text
logits at position 0:
    score for <pad>
    score for the
    score for cat
    score for sat
    score for mat
```

During training, the target tells us which score should be high:

```text
position 0 target = cat
position 1 target = sat
```

During generation, the model chooses a next token from the logits, appends it to
the context, and runs the model again.

## Step 9: Do The Week 01 Task

Your task is to produce one short explanation in your own notes. Use this
structure:

```markdown
# Week 01 Notes

## Decoder-Only In One Sentence

Write one sentence explaining what a decoder-only LLM does.

## The Tiny Example

- text:
- token ids:
- input ids:
- target ids:

## Causal Mask

Answer these two questions:

- What is each position allowed to attend to?
- Why is the future blocked?

## Forward Pass

Write 5-7 sentences tracing this path:

tokens -> embeddings -> causal attention -> MLP -> logits -> loss

## One Thing Still Fuzzy

Write one concept you want to understand better.
```

Keep it short. If you post in Skool, use this format:

```text
Week 01 Submission

Decoder-only in one sentence:

Tiny next-token example:

Why causal masking matters:

My forward-pass explanation:

One thing I want reviewed:
```

## Done Checklist

You are done when:

- you can explain why the target is the input shifted one token to the right
- you can describe what causal masking blocks
- you can trace tokens to logits without naming code files
- your notes explain decoder-only next-token prediction in plain language
- your notes name one thing you still do not understand

Stop there.
