package com.spring.biz.common;

public class AfterAdvice {
	public void afterLog() {
		System.out.println("[사후처리] - 비즈니스로직 수행시 동작");
	}
}
