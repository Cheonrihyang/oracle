package com.spring.biz.common;

import org.aspectj.lang.annotation.Aspect;
import org.aspectj.lang.annotation.Pointcut;

@Aspect
public class PointCutCommon {

	//allPointCut을 메서드로 선언
		@Pointcut("execution(* com.spring.biz..*Impl.*(..))")
		public void allPointCut() {}
		//getPointCut을 메서드로 선언
		@Pointcut("execution(* com.spring.biz..*Impl.get*(..))")
		public void getPointCut() {}
}
