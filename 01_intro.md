# Day 1 - Intro

## Goal
Build a small, honest mental model of a transformer before we write any code.

## Step By Step

1. Start with the big picture.
   - A transformer is a stack of layers that turns token sequences into token predictions.
   - The model does not "understand" text the way a person does.
   - It learns patterns that make the next token more likely.

2. Name the four moving parts.
   - Token embeddings turn ids into vectors.
   - Attention lets tokens look at other tokens.
   - MLP layers transform features after attention.
   - Residual connections keep training stable.

3. Trace one forward pass.
   - Input text becomes token ids.
   - Token ids become embeddings.
   - Attention mixes information across the sequence.
   - The MLP refines the result.
   - The final projection produces logits for the next token.

4. Keep the first implementation tiny.
   - Use a toy vocabulary.
   - Use one attention head.
   - Use one transformer block.
   - Focus on clarity before performance.

5. Decide what success looks like.
   - You can explain the data flow from tokens to logits.
   - You can point to where context is mixed.
   - You can describe why residuals matter.

6. Prepare for the next lesson.
   - In the next step we will build the simplest possible token pipeline.
   - That means embeddings first, then attention, then the full block.

## Run This

- Read this lesson once.
- Sketch the forward pass on paper or in notes.
- Make sure you can say what each block in the model does in one sentence.

## Interview Question

Why do transformers use attention instead of only processing tokens one by one?

