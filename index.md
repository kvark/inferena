---
layout: home
title: Results
---

<div class="tabs">
  <input type="radio" name="model" id="tab-smollm2" value="panel-smollm2" checked>
  <label for="tab-smollm2">SmolLM2-135M</label>
  <input type="radio" name="model" id="tab-smolvla" value="panel-smolvla">
  <label for="tab-smolvla">SmolVLA</label>
</div>

<div id="panel-smollm2" class="tab-content active" markdown="1">
{% include smollm2-135m.md %}
</div>

<div id="panel-smolvla" class="tab-content" markdown="1">
{% include smolvla.md %}
</div>

<ul class="legend">
  <li><strong>Bold</strong> — best among frameworks running the <strong>same model</strong> as PyTorch.</li>
  <li><s>Struck through</s> — framework runs a simplified/different model, not comparable.</li>
  <li><strong>✗</strong> — framework doesn't support this model yet.</li>
  <li>Framework names link to the exact git revision tested.</li>
</ul>

---

**Run it yourself:**

```bash
git clone https://github.com/kvark/infermark && cd infermark
./run.sh                    # all models, all frameworks
./run.sh -m SmolLM2-135M    # single model
```

Results print as markdown tables — paste into the model page and submit a PR.
