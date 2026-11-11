import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

export function formatPercent(val: number | undefined | null): string {
  if (val === undefined || val === null || isNaN(val)) return "N/A";
  return `${(val * 100).toFixed(1)}%`;
}

export function formatDec(val: number | undefined | null, digits: number = 4): string {
  if (val === undefined || val === null || isNaN(val)) return "N/A";
  return val.toFixed(digits);
}

export function formatSpamProbability(prob: number | null | undefined): string {
  if (prob === undefined || prob === null || isNaN(prob)) return "N/A";
  if (prob > 0.9999) return ">99.99%";
  if (prob < 0.0001) return "<0.01%";
  return `${(prob * 100).toFixed(1)}%`;
}

