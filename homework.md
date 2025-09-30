# Problem: Profiling with Nsight Systems (nsys_profile) (5 points)

Profile your forward pass, backward pass, and optimizer step using `nsys` with each of the model sizes described in the provided Table and context lengths of 128, 256, 512, and 1024. You may run out of memory with some of these context lengths for the larger models, in which case just note it in your report.

---

(a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library?

**Deliverable:** A 1-2 sentence response.

<br>

**Answer:**
> *Your answer here.*

---

(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes?

*(Hint: look at the "CUDA GPU Kernel Summary" under "Stats Systems View", and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.)*

**Deliverable:** A 1-2 sentence response.

<br>

**Answer:**
> *Your answer here.*

---
(c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass?

**Deliverable:** A 1-2 sentence response.

<br>

**Answer:**
> *Your answer here.*

---

(d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you'd do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels?

**Deliverable:** A 1-2 sentence response.

<br>

**Answer:**
> *Your answer here.*

---

(e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs?

**Deliverable:** A 1-2 sentence response.

<br>

**Answer:**
> *Your answer here.*