/**
 * Frontend Unit Tests for CV Coach Chat Reducer & Scripted State Machine.
 * Runs directly with Node's built-in test runner: `node --test tests/test_chat_reducer.js`
 */
import test from 'node:test';
import assert from 'node:assert/strict';

// The scripted chat state machine reducer mirroring CVCoachWidget
const initialCoachState = {
  isOpen: false,
  isMinimized: false,
  messages: [],
  isTyping: false,
  bulletIndex: 0,
  activeTone: 'impact',
  isLoadingRewrites: false,
  activeRewrites: null,
  batchProgress: 0,
  batchResults: [],
  jdKeywords: [],
  step: 'idle', // 'idle' | 'greeting' | 'picking_bullet' | 'picking_tone' | 'showing_rewrites' | 'accepted' | 'done'
};

function coachReducer(state, action) {
  switch (action.type) {
    case 'OPEN':
      return { ...state, isOpen: true, isMinimized: false, step: state.step === 'idle' ? 'greeting' : state.step };
    case 'CLOSE':
      return { ...state, isOpen: false };
    case 'TOGGLE_MINIMIZE':
      return { ...state, isMinimized: !state.isMinimized };
    case 'SET_TYPING':
      return { ...state, isTyping: action.payload };
    case 'ADD_MESSAGE':
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: 'm-' + Date.now() + '-' + Math.random().toString(36).substr(2, 4), ...action.payload },
        ],
      };
    case 'START_BULLET_FLOW':
      return {
        ...state,
        bulletIndex: action.payload.index,
        step: 'picking_tone',
        activeRewrites: null,
      };
    case 'SET_TONE':
      return {
        ...state,
        activeTone: action.payload,
        step: 'loading',
      };
    case 'RECEIVE_REWRITES':
      return {
        ...state,
        isLoadingRewrites: false,
        isTyping: false,
        activeRewrites: action.payload,
        step: 'showing_rewrites',
      };
    case 'ACCEPT_REWRITE':
      return {
        ...state,
        step: 'accepted',
      };
    case 'NEXT_BULLET':
      return {
        ...state,
        bulletIndex: action.payload.nextIndex,
        step: action.payload.hasNext ? 'picking_tone' : 'done',
        activeRewrites: null,
      };
    case 'SET_BATCH_PROGRESS':
      return {
        ...state,
        batchProgress: action.payload.progress,
        batchResults: action.payload.results || state.batchResults,
      };
    case 'RESET':
      return { ...initialCoachState, isOpen: state.isOpen };
    default:
      return state;
  }
}

test('CV Coach Reducer: OPEN and CLOSE lifecycle', () => {
  let state = initialCoachState;
  assert.equal(state.isOpen, false);
  assert.equal(state.step, 'idle');

  state = coachReducer(state, { type: 'OPEN' });
  assert.equal(state.isOpen, true);
  assert.equal(state.isMinimized, false);
  assert.equal(state.step, 'greeting');

  state = coachReducer(state, { type: 'TOGGLE_MINIMIZE' });
  assert.equal(state.isMinimized, true);

  state = coachReducer(state, { type: 'TOGGLE_MINIMIZE' });
  assert.equal(state.isMinimized, false);

  state = coachReducer(state, { type: 'CLOSE' });
  assert.equal(state.isOpen, false);
});

test('CV Coach Reducer: messages appending and typing indicator', () => {
  let state = initialCoachState;
  state = coachReducer(state, { type: 'SET_TYPING', payload: true });
  assert.equal(state.isTyping, true);

  state = coachReducer(state, {
    type: 'ADD_MESSAGE',
    payload: { role: 'bot', type: 'text', payload: { text: 'Hello candidate!' } },
  });
  assert.equal(state.messages.length, 1);
  assert.equal(state.messages[0].role, 'bot');
  assert.equal(state.messages[0].payload.text, 'Hello candidate!');

  state = coachReducer(state, { type: 'SET_TYPING', payload: false });
  assert.equal(state.isTyping, false);
});

test('CV Coach Reducer: complete scripted flow state machine', () => {
  let state = initialCoachState;

  // 1. Open coach -> greeting
  state = coachReducer(state, { type: 'OPEN' });
  assert.equal(state.step, 'greeting');

  // 2. Select first weak bullet -> picking_tone
  state = coachReducer(state, { type: 'START_BULLET_FLOW', payload: { index: 0 } });
  assert.equal(state.bulletIndex, 0);
  assert.equal(state.step, 'picking_tone');

  // 3. User chooses tone 'technical' -> loading
  state = coachReducer(state, { type: 'SET_TONE', payload: 'technical' });
  assert.equal(state.activeTone, 'technical');
  assert.equal(state.step, 'loading');

  // 4. API returns 3 rewrites -> showing_rewrites
  const sampleRewrites = [
    { text: 'Architected microservices pipeline reducing latency by [X%].', style: 'Technical Architecture', impact_score: 9 },
    { text: 'Engineered automated deployments across [N] nodes.', style: 'Systems Scale', impact_score: 9 },
    { text: 'Delivered core cloud infrastructure.', style: 'Concise Executive', impact_score: 8 },
  ];
  state = coachReducer(state, { type: 'RECEIVE_REWRITES', payload: sampleRewrites });
  assert.equal(state.step, 'showing_rewrites');
  assert.equal(state.activeRewrites.length, 3);

  // 5. User accepts rewrite -> accepted
  state = coachReducer(state, { type: 'ACCEPT_REWRITE' });
  assert.equal(state.step, 'accepted');

  // 6. User advances to next weak bullet -> picking_tone
  state = coachReducer(state, { type: 'NEXT_BULLET', payload: { nextIndex: 1, hasNext: true } });
  assert.equal(state.bulletIndex, 1);
  assert.equal(state.step, 'picking_tone');

  // 7. Last weak bullet completed -> done
  state = coachReducer(state, { type: 'NEXT_BULLET', payload: { nextIndex: 2, hasNext: false } });
  assert.equal(state.step, 'done');
});

test('CV Coach Reducer: batch progress tracking', () => {
  let state = initialCoachState;
  state = coachReducer(state, {
    type: 'SET_BATCH_PROGRESS',
    payload: { progress: 50, results: [{ id: 'b-1' }] },
  });
  assert.equal(state.batchProgress, 50);
  assert.equal(state.batchResults.length, 1);

  state = coachReducer(state, { type: 'RESET' });
  assert.equal(state.batchProgress, 0);
  assert.equal(state.messages.length, 0);
});
