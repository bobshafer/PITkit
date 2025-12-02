{-# LANGUAGE NamedFieldPuns #-}
{-# LANGUAGE RecordWildCards #-}

module Main where

import Text.Printf (printf)
import Data.Fixed (mod')
import Data.List  (foldl')
import System.Random (Random, StdGen, mkStdGen, randomR)

-- | Simulation parameters
nOscillators :: Int
nOscillators = 8

dt :: Double
dt = 0.05

nSteps :: Int
nSteps = 350  -- ~17.5 units of time

-- | Oscillator state = Φ + its local K-parameters
data Osc = Osc
  { phase       :: !Double  -- current phase φ_i
  , natFreq     :: !Double  -- intrinsic ω_i
  , kCoupling   :: !Double  -- K_i: learned coupling strength
  , kPrefPhase  :: !Double  -- K_i: preferred phase (memory)
  } deriving (Show)

-- | Global PIT parameters (shared μ, ν)
data PIT = PIT
  { mu :: !Double  -- habit strength
  , nu :: !Double  -- plasticity
  } deriving (Show)

-- | Full system state at a time t
data State = State
  { time   :: !Double
  , pit    :: !PIT
  , oscs   :: ![Osc]
  } deriving (Show)

------------------------------------------------------------
-- Initialization
------------------------------------------------------------

-- Simple deterministic spread of natural frequencies around 1.0
-- ω_i in [0.8, 1.2]; phases in [0, 2π)
randomListR :: Random a => (a, a) -> Int -> StdGen -> ([a], StdGen)
randomListR _ 0 g = ([], g)
randomListR range n g =
  let (x, g1) = randomR range g
      (xs, g2) = randomListR range (n - 1) g1
  in (x : xs, g2)

initialStateFrom :: StdGen -> State
initialStateFrom g0 =
  let (phis, g1)  = randomListR (0.0, 2 * pi) nOscillators g0
      (freqs, _)  = randomListR (0.8, 1.2) nOscillators g1
      oscs        = [ Osc { phase      = p
                          , natFreq    = w
                          , kCoupling  = 0.0
                          , kPrefPhase = p
                          }
                    | (p, w) <- zip phis freqs
                    ]
  in State { time = 0.0, pit = initialPIT, oscs = oscs }

initialPIT :: PIT
initialPIT = PIT { mu = 0.01, nu = 0.99 }

------------------------------------------------------------
-- Utilities
------------------------------------------------------------

wrapPhase :: Double -> Double
wrapPhase x =
  let x' = mod' x (2 * pi)
  in if x' < 0 then x' + 2 * pi else x'

-- Compute Kuramoto-style coherence R
coherence :: [Osc] -> Double
coherence os =
  let (sx, sy) = foldl' (\(cx, cy) Osc{..} ->
                           (cx + cos phase, cy + sin phase)
                        ) (0.0, 0.0) os
      r = sqrt (sx * sx + sy * sy) / fromIntegral (length os)
  in r

-- Mean phase (Kuramoto order parameter angle)
meanPhase :: [Osc] -> Double
meanPhase os =
  let (sx, sy) = foldl' (\(cx, cy) Osc{..} ->
                           (cx + cos phase, cy + sin phase)
                        ) (0.0, 0.0) os
  in atan2 sy sx

------------------------------------------------------------
-- One time step update
------------------------------------------------------------

-- | Update a single oscillator given the mean phase and PIT parameters.
-- Returns (updated oscillator, didReinforce) where didReinforce = True if
-- this oscillator reinforced its coupling & contributed to μ, ν update.
updateOsc :: Double -> PIT -> Osc -> (Osc, Bool)
updateOsc meanΦ PIT{mu} osc@Osc{..} =
  let phaseDiff      = meanΦ - phase
      couplingForce  = kCoupling * sin phaseDiff
      dphi           = natFreq + couplingForce * mu
      newPhase       = wrapPhase (phase + dphi * dt)

      movingToward   = (cos phaseDiff) * dphi > 0

      -- Reinforcement / decay of K-coupling
      (newK, reinforced) =
        if movingToward
          then (min 5.0 (kCoupling + 0.1), True)
          else (max 0.0 (kCoupling - 0.02), False)

      newPref = 0.95 * kPrefPhase + 0.05 * meanΦ

      newOsc = osc { phase = newPhase
                   , kCoupling = newK
                   , kPrefPhase = newPref
                   }
  in (newOsc, reinforced)

-- | Update PIT global parameters μ, ν using a logistic-style growth/decay
-- gated by current coherence (higher coherence => faster habit growth).
updatePIT :: PIT -> Double -> PIT
updatePIT PIT{..} coherenceR =
  let growthRate = 0.5   -- tune habit growth speed
      decayRate  = 0.5   -- tune plasticity decay speed
      muDelta    = growthRate * coherenceR * mu * (1 - mu)
      nuDelta    = decayRate  * coherenceR * nu * (1 - nu)
      mu'        = min 0.95 (mu + muDelta)
      nu'        = max 0.02 (nu - nuDelta)
  in PIT { mu = mu', nu = nu' }

-- | Perform one simulation step.
stepState :: State -> State
stepState State{..} =
  let mPhase       = meanPhase oscs
      coh          = coherence oscs
      (oscs', _ )  = unzip (map (updateOsc mPhase pit) oscs)
      pit'         = updatePIT pit coh
      t'           = time + dt
  in State { time = t', pit = pit', oscs = oscs' }

------------------------------------------------------------
-- Simulation & CSV output
------------------------------------------------------------

-- Produce all states from step 0 .. nSteps
simulate :: Int -> State -> [State]
simulate steps s0 = take (steps + 1) $ iterate stepState s0

-- Emit CSV header
csvHeader :: String
csvHeader =
  "step,t,oscillator,phase,naturalFreq,K_couplingStrength,K_preferredPhase,mu,nu,coherence"

-- Convert a state at a given step to CSV lines (one per oscillator)
stateToCSV :: Int -> State -> [String]
stateToCSV stepIdx State{..} =
  let PIT{..} = pit
      coh     = coherence oscs
      idxs    = [0 .. length oscs - 1]
  in [ printf "%d,%.6f,%d,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f,%.9f"
           stepIdx
           time
           idx
           phase
           natFreq
           kCoupling
           kPrefPhase
           mu
           nu
           coh
     | (idx, Osc{..}) <- zip idxs oscs
     ]

main :: IO ()
main = do
  putStrLn csvHeader
  -- Fixed seed for reproducibility; swap mkStdGen for getStdGen to vary runs
  let seed   = mkStdGen 1234
      states = simulate nSteps (initialStateFrom seed)
  mapM_ putStrLn . concat $
    zipWith stateToCSV [0..] states
