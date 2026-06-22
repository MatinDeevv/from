# 50 LinkedIn Posts — `from` / Medallion-Lite

**Read this first (honesty note).** Nobody can *guarantee* a post goes viral — anyone who
promises that is selling something. What you *can* engineer is the stuff virality is made
of: a scroll-stopping first line, one concrete number, a real tension, and a reason to
comment. Every post below is built on that formula and is ready to paste.

**How to use them**
- One per business day ≈ 10 weeks of content.
- The first line is the hook — LinkedIn truncates at ~2 lines, so the hook does all the
  work. Keep the line break after it.
- Don't post all 50 in a week; pace them. Rotate the categories so your feed isn't monotone.
- Swap in a real chart/screenshot where noted `[attach: …]` — posts with one visual
  outperform text-only.
- Replace any number you can't actually defend. Your credibility is the only moat. The
  whole *point* of this project is honest numbers — don't torch that in the marketing.

Categories: 🪝 Hook/contrarian · 🧠 Teach · 🔧 Build-in-public · 📉 Failure/audit ·
🧪 Deep-dive · 🎯 Career/lessons

---

### 1 🪝
I deleted a 100-million-parameter model and replaced it with one that has 97 thousand.

The big one had a Sharpe of 3. The small one had a Sharpe of 0.4.

The small one was the honest one.

Model size doesn't create edge in trading. Four things do: meaningful labels, real features, ensemble diversity, and validation honest enough that the backtest equals reality.

I spent a year learning that the expensive way. Thread on what actually moved the needle 👇

#quant #machinelearning #trading

---

### 2 🪝
"My backtest shows a Sharpe of 4."

Cool. How many configurations did you try before you found it?

"...like 200?"

Then your real Sharpe is probably 0.

The Deflated Sharpe Ratio exists for exactly this reason — it discounts your result by how hard you searched. If you don't deflate, you're not reporting an edge. You're reporting the luckiest of 200 coin flips.

#quantitativefinance #datascience #investing

---

### 3 🧠
The single most expensive bug I've ever shipped didn't crash anything.

It computed normalization stats during training, used them in memory, and never saved them.

So at inference the model got raw, unscaled inputs it had never seen. It "trained perfectly." It would have failed silently with real money.

The lesson: in ML for trading, the dangerous bugs don't throw exceptions. They produce a beautiful, plausible, completely fictional backtest.

#machinelearning #mlops #softwareengineering

---

### 4 📉
Things my trading model audit found that all would have produced a "working" backtest:

– Normalization stats never saved → inference got garbage
– PnL measured in units that don't exist
– A key feature hardcoded to zero
– Standard errors computed on overlapping samples (inflated significance)

Every single one looked fine on a chart.

This is why I trust audits, not dashboards.

#quant #datascience #riskmanagement

---

### 5 🧪
Most retail trading models predict the wrong thing.

They label "did price go up over the next N ticks?" — by the endpoint.

But a real position has a stop and a target. A trade that spikes +3 then bleeds to −1 made money (you'd have exited at the target) but gets labeled a loss.

Triple-barrier labeling fixes this: label by the FIRST barrier touched — take-profit, stop-loss, or time. Now your label matches how money is actually made.

It's the highest-ROI change I've made. Bigger than any model.

#quantitativefinance #machinelearning

---

### 6 🪝
Unpopular opinion: 90% of "AI trading" projects fail before the model is ever trained.

They fail at the label.

If you predict the wrong target, the best transformer on earth will perfectly learn a thing that doesn't make money.

Garbage target in, confident garbage out.

#AI #trading #quant

---

### 7 🔧
Building a tick-level gold prediction engine from scratch in C++/CUDA. No PyTorch.

Why from scratch? Because I wanted to understand every float that touches the model.

Current state:
– 539M XAUUSD ticks
– 27 microstructure features per tick
– ~2M samples/sec on a $250 GPU
– cuBLAS forward/backward, custom Adam kernel

Building in public. Follow along 👇

#cpp #cuda #buildinpublic

---

### 8 🧠
Order Flow Imbalance is the most underrated signal in short-horizon trading.

Most people predict price from past price. OFI predicts price from the *demand for immediacy* in the order book — the signed change in bid/ask sizes.

It measures pressure directly instead of inferring it from history.

If you only add one microstructure feature, add OFI.

#marketmicrostructure #quant #trading

---

### 9 🧪
There's a number called Kyle's Lambda that tells you how much *informed* trading is happening.

It's the slope of price against signed order flow — how much the price moves per unit of net buying.

High lambda = someone who knows something is pushing the market.

It's from a 1985 paper by Albert Kyle. Forty years later it's still one of the cleanest measures of information asymmetry we have.

I run it on a 64-tick rolling window. [attach: lambda time series]

#marketmicrostructure #quantitativefinance

---

### 10 📉
I once had a feature in my model that was hardcoded to 0.0.

For weeks.

The comment said "simplified — expensive, skip for speed." Translation: I never finished it.

So the model spent capacity learning that dimension 21 was always zero. Dead weight, literally.

Audit your features. The ones you forgot about are the ones lying to you.

#machinelearning #datascience #mlops

---

### 11 🪝
Your backtest has 50,000 trades.

How many are independent?

If your samples overlap — and in tick data they always do — the answer might be 5,000.

Every t-stat, every Sharpe, every confidence interval you computed on 50,000 is inflated by √10.

Effective sample size is the difference between a real result and a mirage.

#statistics #quant #datascience

---

### 12 🧠
"Effective sample size" in one example:

You take a photo of a crowd every second for an hour. That's 3,600 photos.

But the crowd barely changed second to second. You don't have 3,600 independent observations of the crowd — you have maybe 60.

Overlapping trading windows are the same. Treat 3,600 like 3,600 and your statistics lie to you.

#statistics #datascience

---

### 13 🔧
Today I cut my model's memory footprint by 40×.

Naively, training on raw 256-tick windows would need ~440 GB — every window materialized.

Instead: keep the tick stream resident on the GPU *once* (~11 GB), and gather each window on the fly in a CUDA kernel from an index.

Same math. 40× less memory. Now it runs on commodity hardware.

#cuda #gpu #performance

---

### 14 📉
Confession: my first backtest's "PnL" was in units that don't exist.

It was |price change| / spread, minus a "cost" of 0.30.

0.30 of what? Dollars? Pips? Spreads? I genuinely couldn't tell you.

The Sharpe ratio I'd been proudly quoting was dimensionless nonsense.

If you can't state the units of your PnL, you don't have a backtest. You have a vibe.

#quant #trading #riskmanagement

---

### 15 🧪
Three families of bugs in financial ML, ranked by how badly they fool you:

1. Crashes — harmless, you fix them immediately.
2. Wrong-number bugs — bad, but a sanity check catches them.
3. Plausible-result bugs — catastrophic. The backtest looks great. The edge is fake. You find out with real money.

99% of effort goes to #1. 99% of the danger is in #3.

#mlops #machinelearning #softwareengineering

---

### 16 🪝
The hardest part of quant trading isn't the model.

It's building a system that refuses to lie to you.

Purging. Embargo. Effective sample size. Deflated Sharpe. A holdout you only look at once.

None of it makes the model smarter. All of it makes the result *true*.

That's the whole game.

#quantitativefinance #datascience

---

### 17 🧠
What "purging" means in walk-forward validation, plainly:

If a training sample's outcome unfolds over ticks that overlap your test period, the model is literally seeing part of the answer.

Purging deletes those overlapping training samples.

Embargo deletes a buffer right after the test window too, because correlation leaks backwards.

Skip these and your validation is cheating. Quietly.

#machinelearning #quant #crossvalidation

---

### 18 🔧
A $250 RTX 3050 is training my model at ~2M samples/second.

The trick isn't the GPU. It's keeping it fed:
– Batch size 8192 (fill the matrix units)
– Triple-buffered pipeline (GPU never waits on data)
– 16 worker threads doing feature extraction
– Prefetch queue depth 20–24

A starved GPU at 88% util beats an idle GPU at 100% spec sheet.

#cuda #gpu #performance #cpp

---

### 19 📉
I replaced a feature in my model and my accuracy went DOWN.

The old feature was the window maximum. The new one was the regression slope.

Max felt informative. But max is dominated by one outlier and tells you nothing about direction.

Slope tells you where price is *going*.

Accuracy went down, but profit factor went up. Because accuracy was never the point.

#machinelearning #quant #datascience

---

### 20 🪝
Accuracy is a vanity metric in trading.

A model that's 52% accurate can be wildly profitable. A model that's 70% accurate can go broke.

What matters: edge per trade, profit factor, and Kelly fraction.

If your trading model's headline metric is accuracy, you're optimizing the wrong thing.

#quant #trading #machinelearning

---

### 21 🧠
The Kelly criterion in one line:

bet fraction = p − (1−p)/b

where p is your win rate and b is your win/loss ratio.

It's the mathematically growth-optimal bet size. John Kelly derived it at Bell Labs in 1956.

The catch nobody mentions: full Kelly assumes you know your edge *exactly*. You don't. So you bet a fraction of Kelly — and sleep at night.

#quantitativefinance #riskmanagement #investing

---

### 22 🧪
Why I bet a *fraction* of Kelly, never full:

Full Kelly is growth-optimal — IF your edge estimate is perfect.

Your edge estimate is never perfect.

And full Kelly on an overestimated edge isn't just suboptimal. It's ruin.

Half-Kelly gives up ~25% of growth for a massive cut in drawdown variance. That trade is almost always worth it.

#riskmanagement #quant #investing

---

### 23 🔧
Build-in-public update: my validation harness now reports 13 columns per fold.

trades · N_eff · winrate · edge · PF · kelly · maxDD · SE · t_stat · sharpe · CI_low · CI_high

The two that matter most? N_eff (are my samples real?) and CI (could this be zero?).

A point estimate without a confidence interval is a guess wearing a suit.

#datascience #quant #statistics

---

### 24 🪝
I have a rule: I look at my final holdout dataset exactly once.

Ever.

The moment you tune anything after seeing it, it stops being a holdout and becomes another validation set you've overfit to.

One look. One number. That number is the truth.

The discipline of *not looking* is harder than any algorithm.

#quant #datascience #machinelearning

---

### 25 🧠
Meta-labeling changed how I think about trading models.

Idea: split the decision in two.

Model 1 decides the DIRECTION (up or down).
Model 2 decides WHETHER TO BET AT ALL, and how much.

Model 1 can be mediocre. Model 2 concentrates your capital where Model 1 is actually reliable and sits out the rest.

It turns a noisy classifier into a selective strategy. López de Prado's best idea, IMO.

#machinelearning #quant #quantitativefinance

---

### 26 📉
Class collapse almost killed my model and I didn't notice for days.

Gold ticks are ~90% "no significant move." So the model learned the optimal lazy strategy: predict NEUTRAL every time. 90% accurate! Zero edge.

The fix: class-weighted loss. Make the rare directional moves *expensive* to miss.

If your model is suspiciously accurate on imbalanced data, it's probably cheating.

#machinelearning #datascience

---

### 27 🧪
27 features. 4 economic stories.

I don't add indicators. I add *estimators of latent quantities*:

– Information asymmetry: OFI, Kyle's lambda
– Liquidity & cost: Amihud, Roll spread, spread compression
– Momentum/reversion: autocorrelation, velocity
– Regime: volatility ratio, volume clock

A model fed only price can't separate these. One fed microstructure can.

#marketmicrostructure #quant

---

### 28 🪝
The most dangerous phrase in quantitative finance:

"It works on the backtest."

Of course it does. You built it on the backtest. You selected it because it worked on the backtest.

The question is never "does it work in-sample." It's "will it work on data I have never, ever touched."

Everything else is decoration.

#quant #datascience #investing

---

### 29 🔧
I write my validation metrics ONCE and share them across every model variant.

The summary model and the deep model literally `#include` the same metrics header.

Why? So I can never accidentally grade two models on two different rulers.

Same code → byte-identical statistics → honest comparison.

Boring engineering. Saves you from lying to yourself.

#softwareengineering #quant #mlops

---

### 30 🧠
Roll's spread is a beautiful trick.

You want the effective bid-ask spread but you only have prices, no quotes.

Roll (1984): the spread shows up as *negative serial covariance* in price changes — the price bounces between bid and ask. Measure the bounce, recover the spread.

Forty years old. Still elegant. Still works.

#marketmicrostructure #quantitativefinance

---

### 31 📉
My GPU kernel had a loop that stepped by 4: `for (k = 0; k < IN; k += 4)`.

IN was 176. 176 ÷ 4 = 44. Clean.

Then I grew the input to 243. 243 ÷ 4 = 60 remainder 3.

The last 3 features were silently dropped. No crash. No error. Just three features quietly thrown away on every forward pass.

Hardcoded constants are landmines. They don't beep.

#cuda #cpp #softwareengineering

---

### 32 🪝
You can't backtest your way to confidence.

You can only audit your way there.

I trust a model with a mediocre Sharpe and a clean audit infinitely more than a model with an amazing Sharpe and no audit.

One of those is a result. The other is a story.

#quant #datascience #riskmanagement

---

### 33 🧪
How I keep 100M ticks on a 4GB GPU:

I don't. I keep them on a bigger one — but the *technique* scales down.

Store the feature stream once as [n_ticks × 27]. Don't materialize windows. Gather each training window on the fly in a kernel from its start index.

Memory: O(ticks), not O(ticks × window). On tick data that's a 40–500× difference.

#cuda #gpu #performance

---

### 34 🧠
Volatility-scaled barriers are why my labels finally made sense.

A 3-pip move means nothing in a news spike and everything in a dead session.

So I stopped using fixed thresholds. Take-profit and stop-loss are set as multiples of *recent realized volatility*. The barriers breathe with the market.

Same model. Honest labels. Night and day.

#quant #machinelearning #trading

---

### 35 🔧
Things I built from scratch in C++ for this project, because I refused to import them:

attention · conv1d · GRU · LSTM · mixture-of-experts · FiLM · gated residual nets · SAM · IRM · gradient surgery · a triple-barrier labeler · a López de Prado sample-weighter

Did I need to? No. Do I now understand every one of them? Yes.

#cpp #machinelearning #buildinpublic

---

### 36 📉
The audit I ran on my own code found 3 critical bugs and 6 architectural weaknesses.

My first reaction was embarrassment.

My second was relief — because every one of those bugs was producing a *convincing* backtest, and I'd nearly trusted it.

An audit isn't an admission of failure. It's the only thing standing between you and a confident, expensive mistake.

#mlops #quant #softwareengineering

---

### 37 🪝
"Compute saturation = depth × breadth."

Depth: big sequence models that reach real dynamics.
Breadth: hundreds of validation fits across folds, seeds, and regimes.

Depth alone overfits one split.
Breadth alone overfits your *search*.

You need both. And then you deflate for how hard you looked.

#quant #machinelearning #datascience

---

### 38 🧠
Why I treat a negative Kelly fraction as a BUG, not a result.

If my model's optimal bet size comes out negative, it's not telling me "the edge is small."

It's telling me "you have negative expectancy" — and in my experience that's almost always a broken normalization pipeline, not a real market signal.

Some metrics are answers. Some are smoke alarms.

#quant #datascience #riskmanagement

---

### 39 🧪
Adversarial validation = make your model fight the dumbest possible opponents.

Before I believe any signal, I pit it against:
– pure momentum ("it went up, buy")
– pure contrarian ("it went up, sell")

…across every session and volatility regime, after costs.

If it can't beat a one-line strategy, it isn't a strategy. It's overfitting with extra steps.

#quant #machinelearning #datascience

---

### 40 🎯
A year ago I thought quant trading was about finding the best model.

Now I know it's about building the most honest measurement system, and letting it tell you — usually — that you don't have an edge.

The models are the easy part. The honesty is the whole job.

#career #quant #lessonslearned

---

### 41 🪝
I built a system whose main feature is that it tries to prove me wrong.

Purges leakage I'd have missed. Discounts significance I'd have inflated. Hides the holdout from me. Pits my signal against idiot baselines.

The best research tool I own is the one that argues with me.

#datascience #quant #machinelearning

---

### 42 🧠
The Amihud illiquidity measure is one division:

|return| / volume

That's it. How much did the price move per unit of volume traded?

High value = thin, fragile market where small orders move price. Low value = deep, liquid market.

The simplest ideas in microstructure are often the most robust. Complexity is not a virtue here.

#marketmicrostructure #quant

---

### 43 🔧
Telemetry I print on every single training step:

data=8ms gpu=15ms total=23ms | steps/sec=43 | queue=23 gpu_q=2 | gpu=88% vram=2.8GB cpu=82%

Why? Because "training is slow" is useless. "The prefetch queue is at 0" is actionable.

Measure the bottleneck, not the symptom.

#performance #cuda #mlops

---

### 44 📉
I once celebrated a model hitting 70% accuracy on imbalanced data.

For about ten minutes.

Then I realized 68% of the data was one class. My "genius" model had learned to say one word.

Now the first thing I check on any classifier is the base rate. Always know what "do nothing" scores before you celebrate "doing something."

#datascience #machinelearning

---

### 45 🧪
The Deflated Sharpe Ratio, intuitively:

Run 1 strategy, get a Sharpe of 2 → impressive.
Run 1,000 strategies, keep the best, get a Sharpe of 2 → expected by pure luck.

DSR corrects your Sharpe for: how many things you tried, how varied they were, and how non-normal your returns are.

It's the antidote to the most common self-deception in quant.

#quantitativefinance #statistics #datascience

---

### 46 🪝
Most people optimize their model.

I spend more time optimizing my *doubt*.

Every guardrail in my system — purging, embargo, effective sample size, deflation, one-shot holdout — exists to make a fake edge harder to believe.

Make the truth easy to find and the lies hard to keep. That's the job.

#quant #datascience #machinelearning

---

### 47 🎯
If you're getting into quant ML, learn these before you learn another architecture:

1. Triple-barrier labeling
2. Sample uniqueness / overlap weighting
3. Purged & embargoed cross-validation
4. Deflated Sharpe
5. Fractional Kelly

Four chapters of López de Prado will protect you more than four years of model-tuning.

#career #quant #machinelearning

---

### 48 🧠
Two-pass normalization saved my model.

Pass 1: normalize raw features online as they stream in (Welford), freeze the stats early so they can't see the future.

Pass 2: z-score the summary vectors and *save those stats into the model file*.

The bug I'd had: I did pass 2 at train time and forgot to save it. Inference got raw inputs. The fix was one serialization line. The impact was the whole model.

#machinelearning #mlops #datascience

---

### 49 🔧
The final report from my whole system is a single number.

After 539M ticks, 27 features, 180 model fits, purging, embargo, and deflation…

…it's one after-cost Sharpe on a dataset I'd never touched.

All that machinery exists to make *one number* trustworthy. That's not inefficiency. That's the point.

#quant #datascience #buildinpublic

---

### 50 🪝
For a year I tried to build a model that makes money.

I failed at that — but I succeeded at something better: I built a system that can tell, honestly, whether ANY model makes money.

The first is a lottery ticket. The second is a printing press for lottery-ticket evaluators.

I'll take the press.

(Open-sourcing the validation harness soon. Comment "harness" if you want the repo.)

#quant #machinelearning #buildinpublic #opensource

---

## Posting playbook (bonus)

- **Cadence:** 3–5×/week. Consistency beats volume.
- **Best openers here:** #1, #2, #28, #50 — strong contrarian hooks, save for when you want reach.
- **Best for credibility with quants:** #5, #9, #17, #25, #45 — they signal you actually know the literature.
- **Best for engineers:** #13, #18, #31, #33, #43.
- **Add a visual** to #9, #13, #18, #23 — charts/telemetry screenshots lift these a lot.
- **Engagement bait that isn't cringe:** end ~1 in 5 posts with a genuine question or a
  "comment X for the repo" (see #50). Don't do it every time.
- **The meta-move:** posts #3, #4, #10, #14, #26, #31, #36, #44, #48 are all *failure*
  stories. Counterintuitively these travel furthest on LinkedIn — vulnerability + a
  concrete technical lesson is the platform's highest-performing shape. Lead with these.
