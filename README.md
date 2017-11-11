## Optimizing Kidney Paired Donation via Reinforcement Learning

<img src="figures/pointer.png" width = 800>

<b>Abstract</b>
We propose to apply recent advances in the field of reinforcement learning to the dynamic kidney exchange problem. We recast it as a Markov decision problem and attempt to derive a policy represent by a pointer network. 

Our objective is twofold: 

+ To see how a dynamic algorithm “chooses” to allocate different matchings over time
+ To investigate how much is gained by acting dynamically.

## Main idea

Most of that literature on organ matching is concentrated on variations of *static* models: that is, they do not take into account the fact that patients and donors may enter or leave the patient-donor pool as time progresses. With notable exceptions, optimal mechanisms for dynamic kidney remains a largely unsolved problem.


So instead of deriving a mechanism analytically, we are proposing the opposite approach: we will formulate the dynamic kidney matching problem as a Markov decision process, and then use recent advances in the field of artificial intelligence and reinforcement learning to produce a central agent that will then search for a solution.

## Benchmarks

At the moment we have only tested our algorithm against a "random" benchmark. Next, we will test again a discrete-time variation of an <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2394319">algorithm</a> designed by Akbarpour, Li, Oveis Gharan (2017). 


