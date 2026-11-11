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
