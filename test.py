from transformers import pipeline
import torch

pipe = pipeline(
    model="bigscience/bloom",
    max_new_tokens=20,
    model_kwargs={
        "device_map": "auto",
        "torch_dtype": torch.bfloat16,
    }
)
# pipe = pipeline(
#     model="bigscience/bigscience-small-testing",
#     max_new_tokens=2,
#     model_kwargs={
#         "device_map": "auto",
#         "torch_dtype": torch.bfloat16,
#     }
# )
# print(pipe.model.device)


# print(pipe("Math exercise - answers:\n34+10=44\n54+20="))
# print(pipe("test", use_cache=False))
print(pipe("""Paper abstracts, especially in Physics, can be hard to understand. Take for example the following:It
 has been recently shown that the astrophysics of reionization can be
extracted from the Lyα forest power spectrum by marginalizing the memory
 of reionization over cosmological information. This impact of cosmic
reionization on the Lyα forest power spectrum can survive cosmological
time scales because cosmic reionization, which is inhomogeneous, and
subsequent shocks from denser regions can heat the gas in low-density
regions to ∼3×10 4 K and compress it to mean-density. Current approach
of marginalization over the memory of reionization, however, is not only
 model-dependent, based on the assumption of a specific reionization
model, but also computationally expensive. Here we propose a simple
analytical template for the impact of cosmic reionization, thereby
treating it as a broadband systematic to be marginalized over for
Bayesian inference of cosmological information from the Lyα forest in a
model-independent manner. This template performs remarkably well with an
 error of ≤6% at large scales k≈0.19 Mpc−1 where the effect of the
memory of reionization is important, and reproduces the broadband effect
 of the memory of reionization in the Lyα forest correlation function,
as well as the expected bias of cosmological parameters due to this
systematic. The template can successfully recover the morphology of
forecast errors in cosmological parameter space as expected when
assuming a specific reionization model for marginalization purposes,
with a slight overestimation of tens of per cent for the forecast errors
 on the cosmological parameters. We further propose a similar template
for this systematic on the Lyα forest 1D power spectrum.In
 simple terms, this abstract just says that""", use_cache=False))
