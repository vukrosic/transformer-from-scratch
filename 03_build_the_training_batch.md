# Lesson 03: Build The Training Batch

Lesson 02 explained the shape story:

```text
inputs:  [B, T]
targets: [B, T]
```

This lesson turns that into code.

You will build the first useful function in a decoder-only transformer project:

```text
get_batch()
```

Its job is simple:

```text
take a long stream of token ids
sample short windows
return input ids and shifted target ids
```

## Step 1: Start With One Token Stream

A language model trains on a long sequence of token ids.

Use this tiny example:

```python
import torch

data = torch.tensor([1, 2, 3, 4, 1, 5, 1, 2, 3], dtype=torch.long)
```

Read it as:

```text
the cat sat on the mat the cat sat
```

The model will not receive the whole stream at once. It will receive small
training windows sampled from the stream.

## Step 2: Choose Batch Size And Context Length

Use:

```python
B = 2  # batch size
T = 4  # context length
```

This means:

```text
return 2 training examples
each example has 4 input token positions
```

The expected shapes are:

```text
x.shape = [2, 4]
y.shape = [2, 4]
```

`x` is the input ids.

`y` is the next-token targets.

## Step 3: Understand One Window Before Batching

If we start at index `0`, the raw window is:

```text
data[0:5] = [1, 2, 3, 4, 1]
```

It has length `T + 1`, not `T`, because we need shifted targets.

The input is:

```text
x = [1, 2, 3, 4]
```

The target is:

```text
y = [2, 3, 4, 1]
```

So every position learns the next token:

```text
1 -> 2
2 -> 3
3 -> 4
4 -> 1
```

That is all `get_batch()` does, repeated for multiple random start positions.

## Step 4: Find Valid Start Positions

If `T = 4`, every sampled example needs `T + 1 = 5` tokens.

So if the data length is 9:

```text
data length = 9
last valid start = 9 - 4 - 1 = 4
```

Valid starts are:

```text
0, 1, 2, 3, 4
```

Why not start at `5`?

Because:

```text
data[5:10]
```

would ask for token index `9`, but the last valid index is `8`.

In code:

```python
max_start = len(data) - T - 1
```

Then sample random starts:

```python
starts = torch.randint(0, max_start + 1, (B,))
```

For example, `starts` might be:

```text
tensor([0, 3])
```

That means:

```text
example 0 starts at token 0
example 1 starts at token 3
```

## Step 5: Build One Input Row

For one start index `i`, the input row is:

```python
x_row = data[i : i + T]
```

If:

```text
i = 0
T = 4
```

then:

```text
x_row = data[0:4] = [1, 2, 3, 4]
```

The target row starts one token later:

```python
y_row = data[i + 1 : i + T + 1]
```

For the same start:

```text
y_row = data[1:5] = [2, 3, 4, 1]
```

The target is not a different dataset. It is the same stream shifted by one.

## Step 6: Stack Rows Into A Batch

For every sampled start index, create one row.

```python
x = torch.stack([data[i : i + T] for i in starts])
y = torch.stack([data[i + 1 : i + T + 1] for i in starts])
```

If:

```text
starts = [0, 3]
```

then:

```text
x[0] = data[0:4] = [1, 2, 3, 4]
y[0] = data[1:5] = [2, 3, 4, 1]

x[1] = data[3:7] = [4, 1, 5, 1]
y[1] = data[4:8] = [1, 5, 1, 2]
```

So the final batch is:

```text
x = [
  [1, 2, 3, 4],
  [4, 1, 5, 1],
]

y = [
  [2, 3, 4, 1],
  [1, 5, 1, 2],
]
```

And the shapes are:

```text
x.shape = [2, 4]
y.shape = [2, 4]
```

## Step 7: Write The Full Function

Here is the full version:

```python
import torch


def get_batch(data: torch.Tensor, batch_size: int, context_length: int):
    max_start = len(data) - context_length - 1
    starts = torch.randint(0, max_start + 1, (batch_size,))

    x = torch.stack([
        data[i : i + context_length]
        for i in starts
    ])

    y = torch.stack([
        data[i + 1 : i + context_length + 1]
        for i in starts
    ])

    return x, y
```

The important lines are:

```python
data[i : i + context_length]
```

This creates the input tokens.

```python
data[i + 1 : i + context_length + 1]
```

This creates the next-token targets.

## Step 8: Run A Tiny Check

Use:

```python
data = torch.tensor([1, 2, 3, 4, 1, 5, 1, 2, 3], dtype=torch.long)

x, y = get_batch(data, batch_size=2, context_length=4)

print(x)
print(y)
print(x.shape, y.shape)
```

You should see:

```text
two rows in x
two rows in y
each row has four token ids
```

The exact token rows may change because the starts are random.

The shapes should always be:

```text
torch.Size([2, 4]) torch.Size([2, 4])
```

## Step 9: Check One Row By Hand

Pick one row from `x` and `y`.

Example:

```text
x row: [4, 1, 5, 1]
y row: [1, 5, 1, 2]
```

Check it position by position:

```text
4 predicts 1
1 predicts 5
5 predicts 1
1 predicts 2
```

If every target is the next token after the input token, the batch is correct.

This habit matters. Before building attention, embeddings, or loss, make sure
the data entering the model is exactly what you think it is.

## Step 10: Know What This Gives The Transformer

The transformer will receive:

```text
x: token ids shaped [B, T]
```

It will produce:

```text
logits shaped [B, T, vocab_size]
```

The loss will compare:

```text
logits[b, t]
```

against:

```text
y[b, t]
```

That means `get_batch()` creates the training question at every position:

```text
given this prefix, which token comes next?
```

## Step 11: Write Your Lesson 03 Notes

Write a short note using this structure:

```markdown
# Lesson 03 Notes

## get_batch Job

In one sentence, what does get_batch return?

## Shapes

B =
T =
x.shape =
y.shape =

## Shift

Why does y start one token after x?

## Manual Check

x row:
y row:

Explain each input-target pair in that row.
```

Keep the note small. The main win is understanding that training data is just
token windows plus a one-token shift.

## Done Checklist

You are done when:

- you can explain why each sampled raw window needs `T + 1` tokens
- you can explain why `x` and `y` both have shape `[B, T]`
- you can point to one row and verify that `y` is `x` shifted left by one target
- you understand that `get_batch()` prepares the questions the model learns from

Stop here. Lesson 04 can build token and position embeddings on top of this batch.
