# Lesson 01: Decoder-Only Transformer Mental Model

This lesson teaches the first idea behind GPT-style language models:

```text
a decoder-only transformer learns to predict the next token from the tokens before it
```

Read this lesson from top to bottom. Do not worry about PyTorch yet. Before you
write the model, you need a clear picture of what the model is trying to do.

By the end, you should be able to explain this path:

```text
text -> token ids -> token embeddings -> RoPE attention -> MLP -> logits -> next token
```

## Step 1: Start With The Job Of The Model

A decoder-only LLM receives a sequence of tokens and predicts what token should
come next.

Example:

```text
input:
the cat sat on the

target next token:
mat
```

That is generation. The model reads the current prefix and predicts one more
token.

During training, the same idea happens at many positions at once. If the text is:

```text
the cat sat
```

the model learns these predictions:

```text
given "the"       predict "cat"
given "the cat"   predict "sat"
```

This is why decoder-only LLMs are called next-token predictors. They do not
receive a separate label file. The labels come from the text itself.

## Step 2: Turn Text Into Token Ids

Neural networks do not directly read text. They read numbers.

Use this tiny vocabulary:

```text
0 = <pad>
1 = the
2 = cat
3 = sat
4 = on
5 = mat
```

The text:

```text
the cat sat
```

becomes:

```text
[1, 2, 3]
```

Each number is a token id. Real tokenizers use larger vocabularies and can split
words into smaller pieces, but the basic idea is the same:

```text
text pieces become integer ids
```

## Step 3: Create Inputs And Targets By Shifting

For next-token prediction, the input and target are almost the same sequence.
The target is shifted one token to the right.

From:

```text
tokens: [1, 2, 3]
words:  [the, cat, sat]
```

we create:

```text
input ids:  [1, 2]
target ids: [2, 3]
```

Read this as:

```text
input position 0 sees "the"       -> target is "cat"
input position 1 sees "the cat"   -> target is "sat"
```

This shift is the heart of language-model training.

The model is not asked:

```text
What is the whole sentence label?
```

It is asked:

```text
At each position, what token comes next?
```

## Step 4: Replace Token Ids With Vectors

A token id is just an index. The model needs a vector it can do math with.

So the model uses an embedding table:

```text
token id 1 -> vector for "the"
token id 2 -> vector for "cat"
token id 3 -> vector for "sat"
```

If the model dimension is 4, a toy embedding could look like this:

```text
"the" -> [0.2, -0.1, 0.7, 0.4]
"cat" -> [0.9,  0.3, 0.1, -0.5]
```

The exact numbers are learned. The important part is:

```text
each token position becomes a vector
```

## Step 5: Add Position Information With RoPE

Token embeddings alone do not tell the model order.

Without position information, these two sequences would contain the same token
vectors:

```text
the cat
cat the
```

Older GPT-style models often added learned position embeddings directly to token
embeddings:

```text
position 0 vector = token_embedding("the") + position_embedding(0)
position 1 vector = token_embedding("cat") + position_embedding(1)
```

In this course, we will use the more modern approach: RoPE, short for rotary
position embeddings.

RoPE does not add a separate position vector to `x` at the start. Instead, it
adds position information inside attention by rotating the query and key vectors
based on their positions.

For now, keep the mental model simple:

```text
token embeddings say what token is here
RoPE helps attention understand where tokens are relative to each other
```

The transformer still starts with one vector per token position. The difference
is that position information enters when attention compares positions.

## Step 6: Understand Why Attention Must Be Causal

Self-attention lets each position read information from other positions.

In a decoder-only LLM, attention is causal:

```text
a position can look left
a position can look at itself
a position cannot look right
```

For:

```text
[the, cat, sat]
```

the allowed attention pattern is:

```text
position 0, "the":
    can look at "the"

position 1, "cat":
    can look at "the", "cat"

position 2, "sat":
    can look at "the", "cat", "sat"
```

The blocked pattern is:

```text
position 0 cannot look at "cat" or "sat"
position 1 cannot look at "sat"
```

This matters because the model is training to predict the future. If position 0
could see `"cat"` while trying to predict `"cat"`, the task would become
cheating.

The causal mask forces the model to learn from the prefix only.

## Step 7: See What A Decoder Block Does

A decoder-only transformer is a stack of decoder blocks.

Each block has two main parts:

```text
1. causal self-attention
2. MLP
```

Self-attention mixes information between token positions.

The MLP transforms each position independently after attention has mixed the
context.

A simplified block looks like:

```text
x = x + causal_self_attention(x)
x = x + mlp(x)
```

The `x + ...` part is a residual connection. It lets the model update the stream
without destroying the previous information.

Think of the residual stream as the model's working memory:

```text
start with token vectors
attention uses RoPE to compare positions
block 1 edits the vectors
block 2 edits the vectors again
block 3 edits them again
```

After many blocks, each position has a richer vector that represents what the
model knows at that point in the sequence.

## Step 8: Turn Final Vectors Into Logits

After the final decoder block, the model still has one vector per position.

To predict the next token, each vector is projected into vocabulary scores.
These scores are called logits.

If the vocabulary has 6 tokens, one position produces 6 logits:

```text
score for <pad>
score for the
score for cat
score for sat
score for on
score for mat
```

For the example:

```text
input ids:  [1, 2]
target ids: [2, 3]
```

the model should learn:

```text
position 0 logits should give a high score to token 2, "cat"
position 1 logits should give a high score to token 3, "sat"
```

The loss measures how far the logits are from the correct targets. Training
adjusts the model weights so the right next token gets a higher score next time.

## Step 9: Trace The Full Forward Pass

Now connect the full path.

For:

```text
text: the cat sat
token ids: [1, 2, 3]
```

training uses:

```text
input ids:  [1, 2]
target ids: [2, 3]
```

The model does:

```text
1. look up token embeddings for [1, 2]
2. pass the vectors through causal decoder blocks
3. use RoPE inside attention so positions are understood
4. project final vectors into vocabulary logits
5. compare logits against target ids [2, 3]
6. use the loss to improve the weights
```

In one sentence:

```text
the model sees a prefix, produces scores for the next token at each position,
and learns when the correct next token receives too low a score
```

## Step 10: Know The Difference Between Training And Generation

Training uses known text and learns from shifted targets:

```text
input ids:  [the, cat]
target ids: [cat, sat]
```

Generation uses the model repeatedly:

```text
start:  "the"
predict "cat"
append: "the cat"
predict "sat"
append: "the cat sat"
```

Training can score many positions in parallel because the full text is known,
but the causal mask prevents future cheating.

Generation usually happens one token at a time because each new token becomes
part of the next input.

## Step 11: Write Your Lesson 01 Notes

Write a short note using this structure:

```markdown
# Lesson 01 Notes

## Decoder-Only LLM In One Sentence

Write one sentence explaining what a decoder-only LLM does.

## Tiny Example

text:
token ids:
input ids:
target ids:

## Causal Mask

What can each position attend to?
Why is the future blocked?

## Forward Pass

Explain this path in 5-7 sentences:

text -> token ids -> token embeddings -> RoPE attention -> MLP -> logits -> loss

## Training vs Generation

What is different between training and generation?

## Still Fuzzy

Write one thing you want to understand better.
```

Keep the notes short. The goal is not to memorize every term. The goal is to
make the first decoder-only transformer picture clear enough that the next
lesson can turn it into code.

## Done Checklist

You are done when:

- you can explain why targets are inputs shifted one token to the right
- you can describe what causal masking blocks
- you can trace text to logits in plain English
- you can explain why training can use full sequences without future cheating
- you wrote one question that still feels unclear

Stop here. Lesson 02 can start turning this mental model into tensors.
