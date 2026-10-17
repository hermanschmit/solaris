// Fallback values for config keys the server requires at import time.
// The simulator never connects to anything, so these are placeholders.
process.env.SESSION_SECRET ||= "sim";
process.env.CONNECTION_STRING ||= "mongodb://localhost:27017/solaris-sim";
process.env.SERVER_URL ||= "http://localhost:3000";
process.env.CLIENT_URL ||= "http://localhost:8080";
process.env.LOGGING_LEVEL ||= "warn";

export {};
