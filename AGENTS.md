This is a repository for research code. As such, the main goals are 
- Flexible and easy-to-understand code
- Fewer abstractions are typically good, unless directly called for
- APIs may break; there is no need for backwards compatibility unless directly specified
- Code should fail loud and early
- Imports are to be done at the top of the file; no in-function imports
- Do not use leading underscores in function names 
- Do not force a line wrap for overly long lines
- If the user wants to plan or discuss an implementation, do not start implementing it. Give a brief summary of your plan
- Do not use getattr inside a class for its own class members. Class members should always exist and be set to a correct value. In the worst case, class members are set to None in the respective ctor