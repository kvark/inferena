---
layout: default
title: Results
---

<div class="tabs">
  <input type="radio" name="model" id="tab-smollm2" value="panel-smollm2" checked>
  <label for="tab-smollm2">SmolLM2-135M</label>
  <input type="radio" name="model" id="tab-smolvla" value="panel-smolvla">
  <label for="tab-smolvla">SmolVLA</label>
</div>

<div id="panel-smollm2" class="tab-panel active" markdown="1">

{% include smollm2-135m.md %}

</div>

<div id="panel-smolvla" class="tab-panel" markdown="1">

{% include smolvla.md %}

</div>

---

**Legend:**
**Bold** = best among matching frameworks.
~~Struck through~~ = different/simplified model.
**✗** = not supported.
Framework names link to the tested git revision.

**Run it yourself:**
`git clone https://github.com/kvark/infermark && cd infermark && ./run.sh`
