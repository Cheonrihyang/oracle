package com.spring.view2.board;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import javax.servlet.http.HttpSession;

import org.springframework.web.servlet.ModelAndView;
import org.springframework.web.servlet.mvc.Controller;

import com.spring.biz.board.BoardVO;
import com.spring.biz.board.impl.BoardDAO;
import com.spring.biz.board.impl.BoardServiceImpl;

public class GetBoardController implements Controller{
	BoardServiceImpl service;
	public void setBoardService(BoardServiceImpl service) {
		this.service = service;
	}
	@Override
	public ModelAndView handleRequest(HttpServletRequest request, HttpServletResponse response) throws Exception {
		System.out.println("getBoard.to 성공");
		
		String seq = request.getParameter("seq");
		
		BoardVO boardVO = new BoardVO();
		boardVO.setSeq(Integer.parseInt(seq));
		
		//BoardDAO boardDAO = new BoardDAO();
		BoardVO board = service.getBoard(boardVO);
		
		ModelAndView mav = new ModelAndView();
		mav.addObject("board", board);		//model
		mav.setViewName("board1/getBoard");	//view
		return mav;
	}

}
