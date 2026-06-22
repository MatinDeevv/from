# Marketing Kit — `from` / Medallion-Lite

Full go-to-market pack. Viral posts **ranked 1→30** by virality potential, plus taglines,
X/Twitter threads, a Show HN post, a launch email, a YouTube/short script, and a bio kit.

> Reality check: ranking = my estimate of reach potential (hook strength × emotion ×
> specificity × shareability × comment-pull). Not a guarantee. Post #1 here ≠ guaranteed
> #1 in your feed. Test, measure, double down on what lands.

---

## Part 1 — The 30 ranked viral posts (best first)

Scoring per post: **H**ook, **E**motion, **S**pecificity, **Sh**areability, **C**omment-pull
(each /10). Total /50.

---

### #1 — Score 47/50  (H10 E9 S9 Sh10 C9)
**Format: contrarian + reveal**

> I spent a year building an AI to predict gold prices.
>
> It failed.
>
> But I accidentally built something more valuable: a system that can prove, honestly, whether ANY trading model actually makes money — before it loses you a dollar.
>
> Turns out 90% of "profitable" backtests are lying. Here's the machinery that catches the lie 👇
>
> #quant #machinelearning #trading

*Why #1:* failure admission + bigger-win pivot + "everyone's lying" + thread promise. Every viral lever at once.

---

### #2 — Score 46/50  (H10 E9 S8 Sh10 C9)
**Format: shocking number**

> Your backtest says 50,000 trades.
>
> Only 5,000 are real.
>
> The other 45,000 are the same trade wearing different hats — overlapping windows sharing the same future. Every Sharpe, every t-stat you computed is inflated by 3x.
>
> This one mistake has bankrupted more quant funds than bad models ever will.
>
> #quant #datascience #statistics

---

### #3 — Score 45/50  (H10 E8 S9 Sh9 C9)
**Format: confession**

> The most expensive bug of my career didn't crash anything.
>
> It just... forgot to save one set of numbers.
>
> My model trained perfectly. Backtested beautifully. And would have lit money on fire in production — because at inference it received raw inputs it had never seen in training.
>
> In ML, the bugs that crash are mercy. The ones that smile at you are murder.
>
> #machinelearning #mlops

---

### #4 — Score 45/50  (H9 E8 S10 Sh9 C9)
**Format: list / teardown**

> 4 bugs I found in my own trading model. All 4 produced a beautiful backtest:
>
> 1. Normalization stats never saved → live model got garbage
> 2. PnL measured in units that don't exist
> 3. A key feature hardcoded to zero (for weeks)
> 4. Significance computed on fake sample sizes
>
> None crashed. All lied. This is why I audit before I trust.
>
> #quant #mlops #datascience

---

### #5 — Score 44/50  (H10 E8 S8 Sh9 C9)
**Format: myth-bust**

> "My model has a Sharpe of 4."
>
> How many configs did you try to get it?
>
> "~200."
>
> Then your real Sharpe is ~0.
>
> If you test 200 strategies and keep the best, the best one is luck, not edge. The Deflated Sharpe Ratio exists to prove it. Most people have never run it. That's the edge.
>
> #quantitativefinance #investing

---

### #6 — Score 44/50  (H9 E9 S8 Sh9 C9)
**Format: identity/philosophy**

> I built a system whose only job is to prove me wrong.
>
> It hides my test data from me. Discounts the significance I want to believe. Pits my "genius" signal against a strategy a child could write.
>
> The best research tool I own is the one that argues with me.
>
> Confidence is built by surviving doubt, not avoiding it.
>
> #datascience #quant

---

### #7 — Score 43/50  (H9 E7 S9 Sh9 C9)
**Format: underdog flex**

> A $250 GPU is training my model at 2 million samples per second.
>
> Not a cluster. Not an H100. A gaming card that fits in a $600 PC.
>
> The secret isn't the hardware. It's never letting it go hungry: batch 8192, triple-buffered pipeline, 16 worker threads feeding it.
>
> A fed budget GPU beats a starved expensive one. Every time.
>
> #cuda #gpu #performance

---

### #8 — Score 43/50  (H9 E8 S8 Sh9 C9)
**Format: counterintuitive truth**

> Accuracy is a lie in trading.
>
> A model that's 52% accurate can print money.
> A model that's 70% accurate can go broke.
>
> If your headline metric is accuracy, you're playing the wrong game. The real ones: edge per trade, profit factor, Kelly fraction.
>
> Stop optimizing for being right. Optimize for being paid.
>
> #quant #trading #machinelearning

---

### #9 — Score 42/50  (H8 E8 S9 Sh8 C9)
**Format: framework drop**

> The 5 things that protect a quant from himself — learn these before another neural net:
>
> 1. Triple-barrier labels
> 2. Sample-overlap weighting
> 3. Purged + embargoed cross-validation
> 4. Deflated Sharpe
> 5. Fractional Kelly
>
> 4 chapters of López de Prado beat 4 years of model tuning.
>
> Save this.
>
> #quant #machinelearning #career

---

### #10 — Score 42/50  (H9 E7 S9 Sh8 C8)
**Format: tiny-cause big-effect**

> I deleted a 100M-parameter model and shipped one with 97 THOUSAND.

> The big one: Sharpe 3.
> The small one: Sharpe 0.4.
>
> The small one was the honest one. The big one was overfit luxury.
>
> In trading, model size is vanity. The label, the features, and honest validation are the whole game.
>
> #machinelearning #quant

---

### #11 — Score 41/50  (H8 E8 S8 Sh8 C9)
**Format: relatable failure**

> I celebrated my model hitting 70% accuracy.
>
> For ten minutes.
>
> Then I checked: 68% of the data was one class. My "genius" model had learned to say a single word and never shut up.
>
> Always know what "do nothing" scores before you celebrate "doing something."
>
> #datascience #machinelearning

---

### #12 — Score 41/50  (H9 E7 S8 Sh8 C8)
**Format: 1-rule discipline**

> I look at my final test data exactly once. Ever.
>
> The second you tune anything after peeking, it's not a holdout anymore — it's just another set you've overfit to.
>
> One look. One number. That number is the truth.
>
> The hardest algorithm in quant is the discipline of NOT looking.
>
> #quant #datascience

---

### #13 — Score 40/50  (H8 E7 S9 Sh8 C8)
**Format: explain-like-I'm-5**

> "Effective sample size," explained:
>
> Photograph a crowd once per second for an hour → 3,600 photos.
>
> But the crowd barely moved. You don't have 3,600 observations. You have ~60.
>
> Overlapping trading windows are identical. Treat 3,600 like 3,600 and your statistics lie with a straight face.
>
> #statistics #datascience

---

### #14 — Score 40/50  (H8 E7 S9 Sh8 C8)
**Format: old-idea-still-wins**

> A 40-year-old equation still measures insider trading better than most AI.
>
> Kyle's Lambda (1985): the slope of price against signed order flow. How much the price moves per unit of net buying.
>
> High lambda = someone who knows something is pushing the market.
>
> Sometimes the edge is in a 1985 paper, not a 2026 transformer.
>
> #marketmicrostructure #quant

---

### #15 — Score 40/50  (H9 E7 S7 Sh8 C9)
**Format: hot take**

> Unpopular opinion: 90% of AI-trading projects fail before training ever starts.
>
> They fail at the label.
>
> Predict the wrong target and the best model on Earth will flawlessly learn something that makes zero dollars.
>
> Garbage target in → confident garbage out.
>
> #AI #trading #quant

---

### #16 — Score 39/50  (H8 E7 S8 Sh8 C8)
**Format: the fix story**

> My labels were lying until I let them breathe.
>
> Old way: "did price move 3 pips?" — fixed threshold. 3 pips is nothing in a news spike, everything in a dead session.
>
> New way: barriers scaled to live volatility. Take-profit and stop move with the market.
>
> Same model. Honest labels. Completely different result.
>
> #quant #machinelearning

---

### #17 — Score 39/50  (H8 E7 S8 Sh8 C8)
**Format: two-model insight**

> The idea that fixed my trading model: stop asking one model to do two jobs.
>
> Model 1 picks the DIRECTION.
> Model 2 decides WHETHER to bet, and how much.
>
> Model 1 can be mediocre. Model 2 puts capital only where Model 1 is trustworthy and sits out the rest.
>
> A noisy classifier becomes a sniper.
>
> #machinelearning #quant

---

### #18 — Score 38/50  (H8 E7 S8 Sh7 C8)
**Format: silent-killer bug**

> One line of code silently deleted 3 features on every prediction:
>
> `for (k = 0; k < IN; k += 4)`
>
> IN used to be 176 (divisible by 4). I grew it to 243. 243 ÷ 4 leaves 3.
>
> Those last 3 features? Gone. No crash. No warning. Just quietly thrown away forever.
>
> Hardcoded constants don't beep when they betray you.
>
> #cuda #cpp #softwareengineering

---

### #19 — Score 38/50  (H8 E7 S8 Sh7 C8)
**Format: memory flex**

> I cut my model's memory needs by 40x without changing the math.
>
> Naive way: materialize every training window → 440 GB. Impossible.
>
> My way: keep the data on the GPU ONCE, and grab each window on the fly in a kernel from an index.
>
> Same result. 40x less memory. Now it runs on hardware you can afford.
>
> #cuda #gpu #performance

---

### #20 — Score 38/50  (H8 E7 S7 Sh8 C8)
**Format: principle**

> The hardest part of quant trading isn't the model. It's building a system that refuses to lie to you.
>
> Purging. Embargo. Effective samples. Deflated Sharpe. A holdout you see once.
>
> None of it makes the model smarter. All of it makes the result TRUE.
>
> Truth is the product. The model is a detail.
>
> #quantitativefinance #datascience

---

### #21 — Score 37/50  (H7 E7 S8 Sh8 C8)
**Format: simplicity wins**

> The best liquidity signal I use is one division:
>
> |price change| / volume
>
> That's Amihud illiquidity. How much does price move per unit traded?
>
> High = thin, fragile market. Low = deep, liquid.
>
> 20 years old. One operator. Still beats half the fancy stuff.
>
> #marketmicrostructure #quant

---

### #22 — Score 37/50  (H8 E6 S8 Sh7 C8)
**Format: build-in-public**

> Building a tick-level gold trading engine from scratch in C++/CUDA. No PyTorch. No shortcuts.
>
> – 539M ticks
> – 27 microstructure features
> – 2M samples/sec on a budget GPU
> – Custom Adam kernel, cuBLAS math
>
> Why from scratch? So I understand every float that touches the model.
>
> Following along? 👇
>
> #cpp #cuda #buildinpublic

---

### #23 — Score 36/50  (H7 E7 S7 Sh7 C8)
**Format: reframe a metric**

> I treat a negative Kelly fraction as a BUG, not a result.
>
> If the math says "bet a negative amount," it's not whispering "small edge." It's screaming "you have negative expectancy" — and that's almost always a broken pipeline, not the market.
>
> Some numbers are answers. Some are smoke alarms. Know which is which.
>
> #quant #riskmanagement

---

### #24 — Score 36/50  (H7 E7 S7 Sh7 C8)
**Format: adversarial test**

> Before I trust any signal, I make it fight the dumbest opponents alive:
>
> – "It went up, so buy" (momentum)
> – "It went up, so sell" (contrarian)
>
> Across every session, after costs.
>
> If my model can't beat a one-line strategy, it's not a strategy. It's overfitting in a nice suit.
>
> #quant #machinelearning

---

### #25 — Score 35/50  (H7 E6 S8 Sh7 C7)
**Format: under-the-hood**

> Order Flow Imbalance is the most underrated signal in short-term trading.
>
> Everyone predicts price from past price.
>
> OFI predicts price from the demand for immediacy in the order book — the signed change in bid/ask sizes. It measures pressure directly instead of guessing from history.
>
> Add one feature? Add this one.
>
> #marketmicrostructure #quant

---

### #26 — Score 35/50  (H8 E6 S6 Sh7 C8)
**Format: career pivot**

> A year ago I thought quant was about finding the best model.
>
> Now I know it's about building the most honest ruler — and letting it tell me, usually, that I don't have an edge.
>
> The models are the easy part. The honesty is the entire job.
>
> #career #quant #lessonslearned

---

### #27 — Score 34/50  (H7 E6 S7 Sh7 C7)
**Format: elegant-trick**

> A beautiful 1984 trick: measure the bid-ask spread when you have NO quotes, only prices.
>
> Roll's insight: the price bounces between bid and ask, leaving a fingerprint — negative serial covariance in price changes.
>
> Measure the bounce → recover the spread.
>
> Elegance ages well.
>
> #marketmicrostructure #quantitativefinance

---

### #28 — Score 34/50  (H7 E6 S7 Sh7 C7)
**Format: throughput receipts**

> What I print on every training step:
>
> `data=8ms gpu=15ms queue=23 gpu=88% vram=2.8GB cpu=82%`
>
> "Training is slow" is useless.
> "The prefetch queue hit 0" is fixable.
>
> Measure the bottleneck, not the symptom. Your logs should point at the problem, not just describe the pain.
>
> #mlops #performance #cuda

---

### #29 — Score 33/50  (H7 E6 S7 Sh6 C7)
**Format: depth×breadth**

> My north star: compute saturation = depth × breadth.
>
> Depth = big models that reach real market dynamics.
> Breadth = hundreds of validation fits across folds, seeds, regimes.
>
> Depth alone overfits one split. Breadth alone overfits your search.
>
> You need both — then you deflate for how hard you looked.
>
> #quant #machinelearning

---

### #30 — Score 33/50  (H7 E6 S6 Sh7 C7)
**Format: full-stack flex**

> Built from scratch in C++, refusing to `import` any of it:
>
> attention · GRU · LSTM · mixture-of-experts · FiLM · triple-barrier labeler · López de Prado sample-weighter · custom CUDA Adam
>
> Did I need to? No.
> Do I now understand every line? Yes.
>
> Sometimes the long way is the only way that teaches.
>
> #cpp #machinelearning #buildinpublic

---

## Part 2 — Taglines / one-liners (for headlines, hero, bio)

1. "Most backtests lie. This one tries to prove itself wrong."
2. "A printing press for honest trading research."
3. "539M ticks. 27 microstructure features. One number you can actually trust."
4. "I didn't build a model that makes money. I built one that knows when a model doesn't."
5. "Quant validation that argues with you."
6. "The edge isn't the model. It's refusing to fool yourself."
7. "Tick-level gold prediction. Audit-grade honesty. Budget-GPU speed."
8. "Where backtests come to get caught."

---

## Part 3 — X / Twitter thread (repackage of post #1)

> 1/ I spent a year building an AI to trade gold.
> It failed.
> But I built something better: a system that proves whether ANY trading model is real — before it costs you a dollar.
> 90% of "profitable" backtests are lying. Here's how to catch them 🧵
>
> 2/ Lie #1: overlapping samples. Your "50,000 trades" might be 5,000 independent ones. Every Sharpe inflated 3x. Fix: effective sample size.
>
> 3/ Lie #2: leakage. If a training sample's outcome overlaps your test window, the model sees the answer. Fix: purge + embargo.
>
> 4/ Lie #3: the search. Try 200 strategies, keep the best → that's luck. Fix: Deflated Sharpe Ratio.
>
> 5/ Lie #4: the label. Predict the wrong target and a perfect model learns to lose money. Fix: triple-barrier, volatility-scaled.
>
> 6/ Lie #5: peeking. Look at your holdout twice and it's overfit. Rule: look ONCE.
>
> 7/ None of this makes the model smarter. All of it makes the result TRUE.
> That's the whole job.
> Full writeup + open source soon. Follow for the drop.

---

## Part 4 — Show HN / Reddit launch post

**Title:** Show HN: A from-scratch C++/CUDA quant engine built to catch its own lying backtests

> I built `from` — a tick-level XAUUSD (gold) direction engine in pure C++/CUDA, no
> PyTorch. 539M ticks, 27 microstructure features (OFI, Kyle's lambda, Amihud, Roll
> spread), trained at ~2M samples/sec on an RTX 3050.
>
> The interesting part isn't the model — it's the validation. It implements the López de
> Prado stack: triple-barrier labels, sample-uniqueness weighting, purged + embargoed
> walk-forward, effective-sample-size standard errors, and a deflated Sharpe that discounts
> for how many configs were tried. There's a one-shot holdout and an adversarial-baseline
> check (momentum/contrarian) before any signal is believed.
>
> I also published the full self-audit: 3 critical bugs + 6 weaknesses I found in my own
> code, every one of which produced a *convincing* backtest. That section is the most
> useful thing in the repo.
>
> Looking for feedback on the validation methodology and the CUDA on-the-fly window gather
> (keeps memory O(ticks) instead of O(ticks×window), ~40x saving).

---

## Part 5 — Launch email (list / newsletter)

**Subject:** I spent a year proving my own trading model wrong

> Hey —
>
> Quick one. I just open-sourced `from`, a quant engine I built from scratch in C++/CUDA.
>
> It predicts short-term gold direction from 539M ticks of order-flow data. But the part
> I'm actually proud of is that it's built to *catch itself lying*: purged validation,
> deflated Sharpe, a holdout I only looked at once.
>
> I also published every bug I found auditing my own code — including one that produced a
> perfect backtest and would have failed silently with real money.
>
> If you've ever trusted a backtest you shouldn't have, this is for you.
>
> → [link]
>
> Reply and tell me which guardrail you've skipped. I've skipped most of them.

---

## Part 6 — Short-form video / YouTube script (60s)

> [0:00] "This trading model has a Sharpe of 4. It's also completely worthless. Let me show you why."
>
> [0:06] "I tried 200 strategies and kept the best one. That's not skill — that's the maximum of 200 coin flips."
>
> [0:14] "Mistake two: my 50,000 trades? Only 5,000 were independent. The rest overlapped, sharing the same future. My significance was inflated 3x."
>
> [0:24] "Mistake three: I peeked at my test data, tuned, peeked again. By then it was just another overfit set."
>
> [0:32] "So I rebuilt the whole thing around one idea: assume the backtest is lying until proven otherwise."
>
> [0:40] "Purged validation. Deflated Sharpe. A holdout I look at exactly once. Adversarial baselines."
>
> [0:50] "The model got less impressive. And finally, for the first time, believable."
>
> [0:56] "Honest beats impressive. Every time. Code's in the description."

---

## Part 7 — Bio / about kit

**Short bio:** Building `from` — a from-scratch C++/CUDA quant engine for gold, obsessed
with one question: is this edge real, or am I fooling myself?

**LinkedIn headline:** Quant systems in C++/CUDA · Building validation that refuses to lie ·
539M ticks, audit-grade honesty

**One-liner intro for podcasts/panels:** "I build trading systems whose main feature is
that they try to prove me wrong."

---

## Part 8 — Distribution checklist

- [ ] Post #1, #2, #3 in week 1 (your strongest — front-load reach).
- [ ] Attach a visual to #7, #14, #19, #28 (chart/telemetry screenshot = +reach).
- [ ] Repost the X thread (Part 3) the same week as LinkedIn #1.
- [ ] Show HN on a Tue–Thu morning US time.
- [ ] Newsletter email the day the repo goes public.
- [ ] Reply to every comment in the first 2 hours — early engagement drives the algorithm.
- [ ] One short-form video (Part 6) → cross-post to LinkedIn, X, YouTube Shorts, TikTok.
- [ ] Pin tagline #1 or #4 as your profile featured line.
