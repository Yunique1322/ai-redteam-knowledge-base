# Application & System

Attacking everything around the model: the application logic that calls it and the infrastructure it runs on. A model can be perfectly robust and the deployment can still fall to ordinary software and supply-chain weaknesses, now with an AI flavor on top.

[Overview of Application & System Components](Overview%20of%20Application%20%26%20System%20Components/) maps the surface, how an AI feature actually gets wired into an app and a host.

[Attacking the Application](Attacking%20The%20Application/) covers the app layer. Model reverse engineering recovers the architecture, parameters, or behavior from the outside. Denial of ML service is resource exhaustion aimed at inference. Insecure integrated components are weaknesses in the glue around the model. Rogue actions are about getting a model-driven agent to do something it was never meant to.

[Attacking the System](Attacking%20the%20System/) drops to the infrastructure. Excessive data handling and insecure storage is data the pipeline leaves exposed. Model deployment tampering is swapping or altering the served artifacts. Vulnerable framework code is the CVEs and unsafe defaults in the ML frameworks themselves.

[MCP](MCP/) is the Model Context Protocol, with a practical introduction and then the security of it: vulnerable servers, outright malicious servers (tool poisoning, rug pulls, tool shadowing), and how to mitigate the MCP-specific risks. The [Skills Assessment](Skills%20Assessment.md) is the end-to-end exercise.

This is the System component of the methodology, and it is where AI security meets plain appsec and supply-chain security. Agentic systems and MCP widen it sharply: the moment a model can call tools, "rogue actions" and a malicious tool server stop being theoretical and turn into real paths to code execution and lateral movement. It pairs naturally with [Prompt Injection](../prompt-injection/), which is often the trigger, and with the [pickle and steganography](../data-poisoning/Pickles%20and%20Steganography/) work, which is the same supply-chain problem from the data side.
