# Shifna — Experience Providers & Data Logistics

## Overview

This module implements the Experience Providers and Experience Pool components for the Uncertainty-Driven Learning Framework.

The implementation is responsible for generating suitable learning experiences from uncertain samples and organizing the selected experiences into a PyTorch-compatible pool.

---

## Components

### 1. BaseProvider

`BaseProvider` is the abstract base class for all experience providers.

It defines the common `provide()` interface that provider implementations must follow.

The interface accepts:

- `sample_idx` — index of the uncertain sample
- `objective` — learning objective associated with the sample

Provider implementations return a list of experience indices.

---

### 2. CounterexampleProvider

`CounterexampleProvider` selects examples from a confused class that can act as suitable boundary or counterexample experiences.

The provider:

- Uses the embedding representation of samples
- Selects samples belonging to the confused class
- Excludes the uncertain sample itself from its true class
- Supports a configurable number of retrieved experiences
- Handles missing candidate classes
- Handles cases where the requested number of samples is larger than the available candidates

---

### 3. RetrievalProvider

`RetrievalProvider` retrieves relevant examples for an uncertain or outlier sample.

The provider:

- Uses embedding similarity
- Searches for examples from the same true class
- Excludes the query sample itself
- Returns the requested number of relevant examples when available
- Returns an empty list when suitable candidates are unavailable

---

### 4. ExperiencePool

`ExperiencePool` maintains the selected experience indices and constructs a PyTorch `Subset` from the original training dataset.

The pool supports:

- Candidate index validation
- Duplicate removal
- Maximum total candidate selection
- Per-class limits for class balancing
- Basic pool statistics
- Retrieving stored indices
- Clearing the pool

The pool is implemented using PyTorch dataset mechanisms.

---

## Edge Cases Handled

The implementation considers the following cases:

- Missing candidate classes
- Empty provider outputs
- Fewer candidates than requested
- More candidates than requested
- Duplicate candidate indices
- Invalid dataset indices
- Class imbalance
- Empty experience pools

---

## Testing

The implementation includes four test files:

### `test_counterexample.py`

Tests:

- Counterexample retrieval
- Returned index validity
- Missing confused class
- Odd number of requested experiences

### `test_retrieval.py`

Tests:

- Retrieval of same-class examples
- Query sample exclusion
- Returned index validity
- Missing same-class candidates

### `test_experience_pool.py`

Tests:

- Normal pool construction
- Maximum sample limitation
- Duplicate handling
- Class balancing
- Pool statistics
- Empty candidate lists

### `test_provider_pool.py`

Tests the integration of:

`RetrievalProvider → CounterexampleProvider → ExperiencePool`

It verifies that provider outputs can be combined and passed to the experience pool.

---

## Requirements

The implementation requires:

- Python
- NumPy
- PyTorch

Dependencies are listed in `requirements.txt`.

Install them using:

```bash
pip install -r requirements.txt